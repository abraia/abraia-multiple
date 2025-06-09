import os
import math
import json
import tempfile
import subprocess

from pathlib import Path


def video_rotation(video_stream):
    side_data = next(filter(lambda s: s.get('rotation'), video_stream.get('side_data_list', [])), None)
    return video_stream.get('tags', {}).get('rotate') or (side_data and side_data.get('rotation')) or 0


def video_size(width, height, rotation):
    if int(rotation or 0) in [90, -90, 270, -270]:
        return height, width
    return width, height


def media_info(path):
    cmd = f"ffprobe -v quiet -print_format json -show_streams {path}"
    output = subprocess.check_output(cmd, shell=True)
    streams = json.loads(output)['streams']
    video_streams = [stream for stream in streams if stream['codec_type'] == 'video']
    audio_streams = [stream for stream in streams if stream['codec_type'] == 'audio']
    width, height = video_size(video_streams[0]['width'], video_streams[0]['height'], video_rotation(video_streams[0]))
    return {
      'width': int(width), 'height': int(height),
      'duration': float(video_streams[0]['duration']),
      'bitrate': int(video_streams[0]['bit_rate']),
      'fps': eval(video_streams[0]['avg_frame_rate']),
      'par': video_streams[0].get('sample_aspect_ratio', '1:1').replace('0:1', '1:1'),
      'audio': len(audio_streams) > 0
    }


def run_ffmpeg(params, stdio=True):
    cmd = f"ffmpeg -v {'warning' if stdio else 'quiet'} -stats -hide_banner -y {params}"
    subprocess.run(cmd, shell=True, capture_output=True, check=True)


# const ffmpeg = (params, stdio=true) => {
#   params = `ffmpeg -v warning -stats -hide_banner -y ${params}`
#   return new Promise((resolve, reject) => {
#     const child = exec(params)
#     if (stdio) child.stdout.on('data', data => process.stdout.write(data))
#     if (stdio) child.stderr.on('data', data => process.stdout.write(data))
#     child.on('close', (code) => {
#       if (code === 0) resolve()
#       reject(code)
#     })
#   })
# }


def create_poster(src, dest, args={}):
    params = f"-ss {args.get('frame', 0)} -i {src} -vf scale={args.get('width', 'iw')}:{args.get('height', 'ih')},format=rgb24 -vframes 1 -q:v 2 {dest}"
    run_ffmpeg(params, stdio=False)
    return dest


def calculate_frame_rate(fps):
    return min(fps, 60) / 2 if fps > 30 else fps


def bitrate_size(ar, size=640*360):
    height = round(math.sqrt(size / ar) / 2) * 2
    width = round((ar * height) / 2) * 2
    return width, height


def estimate_bitrate(src, info, max_duration=180):
    frame_rate = calculate_frame_rate(info['fps'])
    min_bitrate, max_bitrate = frame_rate / 30 * 350000, frame_rate / 30 * 1650000
    width, height = bitrate_size(info['width'] / info['height'])
    output = os.path.join(tempfile.gettempdir(), 'bitrate.mp4')
    cmd = (f"ffmpeg -v quiet -y -to {max_duration} -i {src} -r {frame_rate} -vf scale={width}:{height} "
           f"-crf 21 -preset veryfast -profile:v baseline -pix_fmt yuv420p {output}")
    subprocess.run(cmd, shell=True, capture_output=True, check=True) # Change to use run_ffmpeg
    info = media_info(output)
    info['bitrate'] = max(min(info['bitrate'], max_bitrate), min_bitrate)
    return info


def bitrate_rule(low_variant, high_variant, gain=1, coef=.75):
    bitrate = gain * low_variant['bitrate']
    low_size = low_variant['width'] * low_variant['height']
    high_size = high_variant['width'] * high_variant['height']
    if (high_size >= low_size):
        return round(math.pow(high_size / low_size, coef) * bitrate)
    return round(math.pow(math.pow(bitrate, 1 / coef) / (low_size / high_size), coef))


def calculate_size(info, variant={}):
    max_size = min(info['width'], info['height'], 1440)
    ar = info['width'] / info['height']
    if variant.get('width'):
        variant['height'] = round(variant['width'] / ar / 2) * 2
    else:
        size = variant.get('size', max_size)
        variant['width'] = round(ar * size / 2) * 2 if ar >= 1 else size
        variant['height'] = size if ar >=1 else round(size / ar / 2) * 2
    return variant


def calculate_segment_duration(duration):
    if duration > 40:
        return 6
    if duration > 20:
        return 4
    return 2


def calculate_variants(info, estimation=None, variants=None):
    if not variants:
        variants = [{'key': '1440', 'size': 1440, 'bitrate': 9100000, 'audio': 160000, 'codec': 'h265'},
                    {'key': '1080', 'size': 1080, 'bitrate': 5950000, 'audio': 160000, 'codec': 'h265'},
                    {'key': '1080', 'size': 1080, 'bitrate': 8500000, 'audio': 160000, 'codec': 'h264'},
                    {'key': '720', 'size': 720, 'bitrate': 4650000, 'audio': 128000},
                    {'key': '480', 'size': 480, 'bitrate': 2300000, 'audio': 96000},
                    {'key': '360', 'size': 360, 'bitrate': 1650000, 'audio': 64000}]
        if info['width'] > info['height']:
            variants.append({'key': '240', 'size': 240, 'bitrate': 865000, 'audio': 64000})
    for variant in variants:
        variant = calculate_size(info, variant)
        codec = variant.get('codec', 'h264')
        variant['key'] = variant.get('key') or f"{variant.get('size') or variant['width']}p"
        variant['key'] = f"h{variant['key']}" if codec == 'h265' else variant['key']
        variant['audio'] = variant.get('audio', 128000)
        if not info.get('audio') or variant.get('noaudio'):
            del variant['audio']
        variant['fps'] = variant.get('fps', calculate_frame_rate(info['fps']))
        variant['sd'] = calculate_segment_duration(info.get('duration', 0))
        if estimation:
            gain = 0.7 if codec == 'h265' else 1
            variant['bitrate'] = bitrate_rule(estimation, variant, gain)
    return list(filter(lambda v: info['width'] >= v['width'], variants))


# MP4

def params_scale(params):
    return f"-vf scale={params['width']}:{params['height']} "


def params_codec(params):
    [bitrate, codec, crf] = [params.get('bitrate'), params.get('codec', 'h264'), params.get('crf', 21)]
    [maxrate_ratio, bufsize_ratio] = [0.9, 1.0] # [1.07, 1.5]
    cmd = ""
    if codec == 'h265':
        cmd = f"-c:v libx265 -profile:v main -tag:v hvc1 -crf {crf} "
    else:
        cmd = f"-c:v libx264 -profile:v high -level 4.1 -crf {crf} "
    if bitrate:
        cmd += f"-b:v {bitrate} -maxrate {round(maxrate_ratio * bitrate)} -bufsize {round(bufsize_ratio * bitrate)} "
    return cmd


def params_audio(params):
    audio = params.get('audio')
    return f"-c:a aac -b:a {audio} " if audio else '-an '


def create_mp4(src, folder, variants=[]):
    Path(folder).mkdir(parents=True, exist_ok=True)
    for variant in variants:
        params = (f"-i {src} -pix_fmt yuv420p -movflags +faststart -preset veryfast -r {variant.get('fps', 30)} "
                  + params_scale(variant) + params_codec(variant) + params_audio(variant)
                  + os.path.join(folder, f"{variant['key']}.mp4"))
        print(params)
        if (params):
            run_ffmpeg(params)


# HLS

def params_hls(folder, variants, container='ts', name=''):
    [maxrate_ratio, bufsize_ratio] = [0.9, 1.0] # [1.07, 1.5] Change to use params_bitrate
    stream_map = ''
    for k, variant in enumerate(variants):
        [key, audio] = [variant['key'], variant.get('audio')]
        stream_map += f"v:{k}" + (f",a:{k}" if audio else '') + f",name:{key} "
    stream_map = stream_map.rstrip()
    params = ''
    for k, variant in enumerate(variants):
        print(variant)
        [width, height, bitrate, audio, codec] = [variant['width'], variant['height'], variant['bitrate'], variant.get('audio'), variant.get('codec')]
        params += (f"-map 0:v -s:v:{k} {width}:{height} -b:v:{k} {bitrate} -maxrate:v:{k} {round(maxrate_ratio * bitrate)} -bufsize:v:{k} {round(bufsize_ratio * bitrate)} " 
                   + (f"-c:v:{k} libx265 -profile:v:{k} main -tag:v:{k} hvc1 " if codec == 'h265' else f"-c:v:{k} libx264 -profile:v:{k} high -level 4.1 ") 
                   + (f"-map 0:a -c:a:{k} aac -b:a:{k} {audio} " if audio else ''))
        [fps, sd] = [variant.get('fps', 30), variant.get('sd', 2)]
    params += (f"-pix_fmt yuv420p -movflags +faststart -preset veryfast -r {fps} -g {2 * fps} -keyint_min {fps} -sc_threshold 0 -crf 21 -ar 48000 -ac 2 -hls_playlist_type vod -hls_time {sd} " 
               + (f"-hls_segment_type fmp4 -hls_flags single_file " if container == 'fmp4' else '')
               + f'-f hls -var_stream_map "{stream_map}" -master_pl_name {name or "master"}.m3u8 {os.path.join(folder, "%v.m3u8")}')
    print(params)
    return params


def params_subtitles(folder, lang, segment_duration=2, container='ts'):
    return (f"-c:s webvtt -f segment -segment_time {segment_duration} "
            f"-segment_list_size 0 -segment_list {folder}/sub_{lang}.m3u8 -segment_format webvtt {folder}/{lang}%d.vtt")


# ERROR: Missing CODECS argument
def create_master_playlist(dest, variants, subtitles=[]):
    txt = '#EXTM3U\n#EXT-X-VERSION:3\n'
    if len(subtitles):
        txt = '#EXTM3U\n'
        for subtitle in subtitles:
            lang = subtitle['lang']
            txt += f'#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID="subs",NAME="{lang.upper()}",LANGUAGE="{lang}",URI="sub_{lang}.m3u8"\n'
    for variant in variants:
        [size, width, height, bitrate] = [variant['size'], variant['width'], variant['height'], variant['bitrate']]
        if len(subtitles):
            txt += f'#EXT-X-STREAM-INF:BANDWIDTH={bitrate},RESOLUTION={width}x{height},SUBTITLES="subs"\n{size}p.m3u8\n'
        txt += f'#EXT-X-STREAM-INF:BANDWIDTH={bitrate},RESOLUTION={width}x{height}\n{size}p.m3u8\n'
    with open(dest, 'w') as f:
        f.write(txt)


def create_playlists(folder, variants, subtitles=[], name=''):
    create_master_playlist(os.path.join(folder, f"{name or 'master'}_h.m3u8"), list(filter(lambda v: v['size'] < 1440, variants)), subtitles)
    create_master_playlist(os.path.join(folder, f"{name or 'master'}_m.m3u8"), list(filter(lambda v: v['size'] < 1080, variants)), subtitles)
    create_master_playlist(os.path.join(folder, f"{name or 'master'}_l.m3u8"), list(filter(lambda v: v['size'] < 720, variants)))


def parse_master_playlist(folder, variants):
    sections = []
    with open(os.path.join(folder, 'master.m3u8')) as file:
        section = ''
        for line in file:
            if line.startswith('#EXT-X-STREAM-INF:'):
                sections.append(section)
                section = ''
            section = section + line
        sections.append(section)
    header = sections[0]
    streams = {}
    sizes = [variant['size'] for variant in variants]
    for size, section in zip(sizes, sections[1:]):
        streams[size] = section
    return header, streams


def build_master_playlist(header, streams, variants, dest):
    playlist = header
    for variant in variants:
        size = variant['size']
        playlist += streams[size]
    with open(dest, 'w') as f:
        f.write(playlist)
    return playlist


def create_master_playlists(folder, variants, name):
    header, streams = parse_master_playlist(folder, variants)
    dest = os.path.join(folder, f"{name or 'master'}")
    build_master_playlist(header, streams, list(filter(lambda v: v['size'] < 1440, variants)), f"{dest}_h.m3u8")
    build_master_playlist(header, streams, list(filter(lambda v: v['size'] < 1080, variants)), f"{dest}_m.m3u8")
    build_master_playlist(header, streams, list(filter(lambda v: v['size'] < 720, variants)), f"{dest}_l.m3u8")


def create_hls(src, folder, variants, subtitles=[], name=''):
    Path(folder).mkdir(parents=True, exist_ok=True)
    variants = list(filter(lambda var: var.get('codec') != 'h265', variants))
    run_ffmpeg(f"-i {src} {params_hls(folder, variants, 'fmp4', name)}")
    segment_duration = variants[0].get('sd', 2)
    for subtitle in subtitles:
        [src, lang] = [subtitle['src'], subtitle['lang']]
        run_ffmpeg(f"-i {src} {params_subtitles(folder, lang, segment_duration, 'fmp4')}")
    create_playlists(folder, variants, subtitles, name)
    create_master_playlists(folder, variants, name)
