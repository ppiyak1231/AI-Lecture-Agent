import os
import re
import uuid
import subprocess
from pathlib import Path
from typing import List
from PIL import Image, ImageDraw, ImageFont

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def ffprobe_duration(path: str) -> float:
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", path
        ]).decode().strip()
        return float(out)
    except Exception:
        # Fallback if ffprobe is not available or errors out
        return 10.0

def generate_fallback_image(title: str, texts: List[str], slide_no: int, work_dir: Path) -> str:
    """Windows/가상환경에서 soffice가 없을 때 Pillow로 세련된 슬라이드 플레이스홀더 이미지를 생성합니다."""
    width, height = 1920, 1080
    
    # 세련된 다크 브라운 & 베이지 그라데이션 느낌의 단색 배경 생성
    image = Image.new("RGB", (width, height), color="#2A2421")
    draw = ImageDraw.Draw(image)
    
    # 둥근 모서리의 테두리 그리기
    draw.rounded_rectangle([40, 40, width - 40, height - 40], radius=24, outline="#D9CDC4", width=3)
    
    # 폰트 로드 시도 (없으면 기본 폰트)
    font_title = None
    font_body = None
    try:
        font_title = ImageFont.truetype("malgun.ttf", 64)
        font_body = ImageFont.truetype("malgun.ttf", 36)
    except IOError:
        try:
            font_title = ImageFont.truetype("NanumGothic.ttf", 64)
            font_body = ImageFont.truetype("NanumGothic.ttf", 36)
        except IOError:
            font_title = ImageFont.load_default()
            font_body = ImageFont.load_default()
            
    # 제목 그리기
    title_text = title if title else f"Slide {slide_no}"
    try:
        draw.text((120, 120), title_text, fill="#E6DFD5", font=font_title)
    except Exception:
        draw.text((120, 120), title_text, fill="#E6DFD5")
        
    # 데코레이션 라인
    draw.line([120, 220, 600, 220], fill="#C4A882", width=4)
    
    # 슬라이드 번호 (우측 하단)
    try:
        draw.text((width - 200, height - 120), f"PAGE {slide_no:02d}", fill="#8C7D72", font=font_body)
    except Exception:
        draw.text((width - 200, height - 120), f"PAGE {slide_no:02d}", fill="#8C7D72")
        
    # 본문 텍스트 렌더링
    y_pos = 280
    max_lines = 8
    line_count = 0
    for text in texts:
        if line_count >= max_lines:
            break
        short_text = text if len(text) < 65 else text[:62] + "..."
        try:
            draw.text((120, y_pos), f"•  {short_text}", fill="#FAF8F5", font=font_body)
        except Exception:
            draw.text((120, y_pos), f"- {short_text}", fill="#FAF8F5")
        y_pos += 70
        line_count += 1
        
    out_path = work_dir / f"slide_fallback_{slide_no}.png"
    image.save(out_path)
    return str(out_path)

def export_slide_as_png(state: dict, idx: int, title: str, texts: List[str]) -> str:
    """슬라이드를 이미지로 변환합니다. soffice가 실패하거나 없는 경우 Pillow 폴백 이미지로 전환합니다."""
    work_dir = Path(state["work_dir"]).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    
    pptx = Path(state["pptx_path"]).expanduser().resolve()
    page_no = idx + 1
    out_prefix = work_dir / "slide_img"
    
    # 1. LibreOffice를 이용한 변환 시도
    lo_profile = f"file:///tmp/lo_profile_{uuid.uuid4().hex}"
    env = os.environ.copy()
    env.update({
        "LANG": "ko_KR.UTF-8",
        "LC_ALL": "ko_KR.UTF-8",
        "SAL_USE_VCLPLUGIN": "gen",
    })
    
    cmd = [
        "xvfb-run", "-a",
        "soffice", "--headless", "--nologo", "--nofirststartwizard", "--norestore",
        f"-env:UserInstallation={lo_profile}",
        "--convert-to", "pdf",
        "--outdir", str(work_dir),
        str(pptx),
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=15)
        pdf_path = work_dir / f"{pptx.stem}.pdf"
        
        if pdf_path.exists():
            ppm_cmd = [
                "pdftoppm",
                "-f", str(page_no), "-l", str(page_no),
                "-png", "-r", "150",
                str(pdf_path),
                str(out_prefix)
            ]
            subprocess.run(ppm_cmd, capture_output=True, text=True, env=env, timeout=10)
            
            png_candidates = list(work_dir.glob(f"slide_img-{page_no}.png"))
            if not png_candidates:
                png_candidates = list(work_dir.glob(f"slide_img-*.png"))
                
            if png_candidates:
                newest_png = max(png_candidates, key=lambda p: p.stat().st_mtime)
                return str(newest_png)
    except Exception:
        pass
        
    # 2. 로컬 윈도우 환경 또는 soffice 미동작 시 Pillow 폴백
    return generate_fallback_image(title, texts, page_no, work_dir)

def render_mp4(image_path: str, audio_path: str, out_mp4: str, width=1920, height=1080) -> bool:
    """오디오와 정지 이미지를 결합하여 MP4 비디오 트랙을 만듭니다."""
    dur = ffprobe_duration(audio_path)
    vf = (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
          f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black")
          
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", image_path,
        "-i", audio_path,
        "-t", str(dur),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        out_mp4
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception:
        return False

def concat_videos_ffmpeg(video_paths: List[str], out_path: str):
    """모든 개별 비디오 조각들을 단일 MP4 파일로 Concat 병합합니다."""
    list_path = out_path + ".txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for v in video_paths:
            f.write(f"file '{os.path.abspath(v)}'\n")
            
    cmd = [
        "ffmpeg", "-y", "-safe", "0", "-f", "concat",
        "-i", list_path, "-c:v", "libx264", "-preset", "veryfast",
        "-c:a", "aac", "-b:a", "192k", out_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except Exception:
        # Fallback to copy concat
        cmd_copy = ["ffmpeg", "-y", "-safe", "0", "-f", "concat", "-i", list_path, "-c", "copy", out_path]
        subprocess.run(cmd_copy, check=True)
