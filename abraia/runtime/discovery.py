"""Discover local cameras, GenICam devices, and ONVIF streams."""

from http.client import HTTPException
import socket
import time
import uuid
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit, urlunsplit
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


DISCOVERY_ADDRESS = ("239.255.255.250", 3702)
DISCOVERY_TIMEOUT = 3.0
SOAP_NAMESPACE = "http://www.w3.org/2003/05/soap-envelope"
NETWORK_NAMESPACE = "http://www.onvif.org/ver10/network/wsdl"
DEVICE_NAMESPACE = "http://www.onvif.org/ver10/device/wsdl"
MEDIA_NAMESPACE = "http://www.onvif.org/ver10/media/wsdl"
SCHEMA_NAMESPACE = "http://www.onvif.org/ver10/schema"
ADDRESSING_NAMESPACE = "http://schemas.xmlsoap.org/ws/2004/08/addressing"
DISCOVERY_NAMESPACE = "http://schemas.xmlsoap.org/ws/2005/04/discovery"
CAMERA_SCAN_LIMIT = 10


def discover_streams():
    """Find usable RTSP URLs advertised by ONVIF cameras on the local network."""
    urls = []
    for device in _discover_onvif_devices():
        try:
            urls.extend(_onvif_stream_urls(device["xaddr"]))
        except ValueError:
            continue
    return list(dict.fromkeys(urls))


def discover_cameras(max_index=CAMERA_SCAN_LIMIT):
    """Probe local camera indices and return the ones OpenCV can open."""
    import cv2

    cameras = []
    for index in range(max_index):
        capture = cv2.VideoCapture(index)
        try:
            if capture.isOpened():
                cameras.append({
                    "source": str(index),
                    "label": f"Camera {index}",
                })
        finally:
            capture.release()
    return cameras


def discover_genicam_devices():
    """List GenICam devices exposed by configured GenTL producers."""
    from ..media.sources.genicam import find_genicam_producers

    try:
        from harvesters.core import Harvester
    except ImportError as error:
        raise ImportError(
            "GenICam discovery requires the optional 'harvesters' package. "
            "Install Abraia with 'pip install abraia[genicam]'."
        ) from error

    producer_files = find_genicam_producers()
    if not producer_files:
        raise RuntimeError(
            "No GenTL Producer was found. Set GENICAM_GENTL64_PATH to the "
            "vendor CTI file or its directory."
        )

    harvester = Harvester()
    try:
        for producer_file in producer_files:
            harvester.add_file(str(producer_file))
        harvester.update()
        devices = []
        for index, info in enumerate(harvester.device_info_list):
            serial = str(getattr(info, "serial_number", "") or "").strip()
            name = str(
                getattr(info, "display_name", "")
                or getattr(info, "model", "")
                or getattr(info, "id_", "")
                or f"GenICam device {index}"
            )
            label = (
                name if not serial or serial in name
                else f"{name} ({serial})"
            )
            device = {
                "source": serial or str(index),
                "label": label,
            }
            devices.append(device)
        return devices
    finally:
        try:
            harvester.reset()
        except Exception:
            pass


def _discover_onvif_devices():
    """Find ONVIF cameras advertising on the local network."""
    devices = {}
    name_prefix = "onvif://www.onvif.org/name/"
    with socket.socket(
        socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP
    ) as sock:
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
        sock.bind(("", 0))
        sock.sendto(_discovery_probe().encode("utf-8"), DISCOVERY_ADDRESS)
        deadline = time.monotonic() + DISCOVERY_TIMEOUT
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            sock.settimeout(min(0.25, remaining))
            try:
                response, _address = sock.recvfrom(65535)
            except socket.timeout:
                continue
            try:
                root = ET.fromstring(response)
            except ET.ParseError:
                continue
            for match in _elements(root, "ProbeMatch"):
                xaddrs = _text(match, "XAddrs").split()
                scopes = _text(match, "Scopes")
                name = next(
                    (
                        unquote(scope[len(name_prefix):])
                        for scope in scopes.split()
                        if scope.startswith(name_prefix)
                    ),
                    "",
                )
                for xaddr in xaddrs:
                    parts = urlsplit(xaddr)
                    if (
                        parts.scheme not in ("http", "https")
                        or not parts.hostname
                        or parts.username
                        or parts.password
                    ):
                        continue
                    path = (
                        parts.path
                        if parts.path not in ("", "/")
                        else "/onvif/device_service"
                    )
                    xaddr = urlunsplit(
                        (parts.scheme, parts.netloc, path, parts.query, "")
                    )
                    devices.setdefault(
                        xaddr,
                        {
                            "name": name or parts.hostname,
                            "xaddr": xaddr,
                        },
                    )
    return list(devices.values())


def _onvif_stream_urls(device_url):
    """Return RTSP URLs for an ONVIF device's media profiles."""
    capabilities_request = ET.Element(
        _tag(DEVICE_NAMESPACE, "GetCapabilities")
    )
    ET.SubElement(
        capabilities_request, _tag(DEVICE_NAMESPACE, "Category")
    ).text = "All"
    capabilities = _soap_request(
        device_url,
        f"{DEVICE_NAMESPACE}/GetCapabilities",
        capabilities_request,
    )
    media = next(_elements(capabilities, "Media"), None)
    media_url = _text(media, "XAddr") if media is not None else ""
    if not media_url:
        return []
    media_parts = urlsplit(media_url)
    if (
        media_parts.scheme not in ("http", "https")
        or not media_parts.hostname
        or media_parts.username
        or media_parts.password
    ):
        return []

    profiles_response = _soap_request(
        media_url,
        f"{MEDIA_NAMESPACE}/GetProfiles",
        ET.Element(_tag(MEDIA_NAMESPACE, "GetProfiles")),
    )
    profiles = [
        profile.get("token", "")
        for profile in _elements(profiles_response, "Profiles")
        if profile.get("token")
    ]
    if not profiles:
        return []

    streams = []
    for token in profiles:
        request = ET.Element(_tag(MEDIA_NAMESPACE, "GetStreamUri"))
        setup = ET.SubElement(request, _tag(MEDIA_NAMESPACE, "StreamSetup"))
        ET.SubElement(setup, _tag(SCHEMA_NAMESPACE, "Stream")).text = (
            "RTP-Unicast"
        )
        transport = ET.SubElement(
            setup, _tag(SCHEMA_NAMESPACE, "Transport")
        )
        ET.SubElement(
            transport, _tag(SCHEMA_NAMESPACE, "Protocol")
        ).text = "RTSP"
        ET.SubElement(request, _tag(MEDIA_NAMESPACE, "ProfileToken")).text = token
        try:
            response = _soap_request(
                media_url,
                f"{MEDIA_NAMESPACE}/GetStreamUri",
                request,
            )
        except ValueError:
            continue
        uri = _text(response, "Uri")
        if uri:
            streams.append(uri)
    return streams


def _discovery_probe():
    message_id = f"urn:uuid:{uuid.uuid4()}"
    return (
        f'<e:Envelope xmlns:e="{SOAP_NAMESPACE}" '
        f'xmlns:w="{ADDRESSING_NAMESPACE}" '
        f'xmlns:d="{DISCOVERY_NAMESPACE}" '
        f'xmlns:dn="{NETWORK_NAMESPACE}">'
        f"<e:Header><w:MessageID>{message_id}</w:MessageID>"
        "<w:ReplyTo><w:Address>"
        "http://schemas.xmlsoap.org/ws/2004/08/addressing/role/anonymous"
        "</w:Address></w:ReplyTo>"
        "<w:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>"
        f"<w:Action>{DISCOVERY_NAMESPACE}/Probe</w:Action></e:Header>"
        "<e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types>"
        "</d:Probe></e:Body></e:Envelope>"
    )


def _soap_request(endpoint, action, body):
    envelope = ET.Element(_tag(SOAP_NAMESPACE, "Envelope"))
    soap_body = ET.SubElement(envelope, _tag(SOAP_NAMESPACE, "Body"))
    soap_body.append(body)
    request = Request(
        endpoint,
        data=ET.tostring(envelope, encoding="utf-8", xml_declaration=True),
        headers={
            "Content-Type": f'application/soap+xml; charset=utf-8; action="{action}"'
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=5) as response:
            payload = response.read()
    except HTTPError as error:
        payload = error.read()
        fault = _fault_message(payload)
        if fault:
            raise ValueError(fault) from error
        if error.code == 401:
            raise ValueError("ONVIF device requires authentication.") from error
        raise ValueError(f"ONVIF request failed (HTTP {error.code}).") from error
    except (URLError, HTTPException, TimeoutError, OSError) as error:
        raise ValueError(f"Could not reach ONVIF device: {error}") from error

    try:
        result = ET.fromstring(payload)
    except ET.ParseError as error:
        raise ValueError(
            "The ONVIF device returned an invalid SOAP response."
        ) from error
    fault = _fault_message(result)
    if fault:
        raise ValueError(fault)
    return result


def _fault_message(payload):
    if isinstance(payload, bytes):
        try:
            payload = ET.fromstring(payload)
        except ET.ParseError:
            return ""
    fault = next(_elements(payload, "Fault"), None)
    if fault is None:
        return ""
    return (
        _text(fault, "Text")
        or _text(fault, "faultstring")
        or "The ONVIF device returned a SOAP fault."
    )


def _elements(root, name):
    if root is None:
        return ()
    return (
        item for item in root.iter()
        if item.tag.rsplit("}", 1)[-1] == name
    )


def _text(root, name):
    element = next(_elements(root, name), None)
    return (element.text or "").strip() if element is not None else ""


def _tag(namespace, name):
    return f"{{{namespace}}}{name}"


__all__ = [
    "discover_cameras",
    "discover_genicam_devices",
    "discover_streams",
]
