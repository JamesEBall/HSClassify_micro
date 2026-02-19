"""Generate sample trade document images for testing OCR."""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import sys

OUT_DIR = Path(__file__).parent.parent / "data" / "sample_documents"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def create_invoice(filename: str, lines: list[str], title: str = "COMMERCIAL INVOICE"):
    """Create a simple invoice-style image."""
    W, H = 800, 600
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    # Use default font (Pillow built-in)
    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        font_body = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except Exception:
        font_title = ImageFont.load_default()
        font_body = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Header
    draw.rectangle([(0, 0), (W, 60)], fill="#2c3e50")
    draw.text((20, 15), title, fill="white", font=font_title)

    # Border
    draw.rectangle([(10, 70), (W - 10, H - 10)], outline="#bdc3c7", width=2)

    # Table header
    draw.line([(10, 110), (W - 10, 110)], fill="#bdc3c7", width=1)
    draw.text((20, 85), "Item Description", fill="#2c3e50", font=font_body)
    draw.text((500, 85), "HS Code", fill="#2c3e50", font=font_body)
    draw.text((650, 85), "Qty", fill="#2c3e50", font=font_body)

    # Body lines
    y = 125
    for line in lines:
        parts = line.split("|")
        desc = parts[0].strip()
        hs = parts[1].strip() if len(parts) > 1 else ""
        qty = parts[2].strip() if len(parts) > 2 else ""
        draw.text((20, y), desc, fill="#333", font=font_body)
        draw.text((500, y), hs, fill="#666", font=font_body)
        draw.text((650, y), qty, fill="#666", font=font_body)
        y += 30
        draw.line([(10, y - 5), (W - 10, y - 5)], fill="#ecf0f1", width=1)

    # Footer
    draw.text((20, H - 40), "Exported for customs clearance", fill="#999", font=font_small)

    img.save(OUT_DIR / filename)
    print(f"Created: {OUT_DIR / filename}")


# English invoice
create_invoice("invoice_en.png", [
    "Fresh boneless beef cuts | 020130 | 500 kg",
    "Frozen shrimp, peeled | 030617 | 200 kg",
    "Thai jasmine rice, milled | 100630 | 1000 kg",
    "Lithium-ion battery packs | 850760 | 50 units",
    "Cotton T-shirts, men's knitted | 610910 | 2000 pcs",
    "Laptop computers 14 inch | 847130 | 100 units",
])

# Thai packing list
create_invoice("packing_list_th.png", [
    "ข้าวหอมมะลิไทย ขัดสี 5% หัก | 100630 | 25 ตัน",
    "กุ้งแช่แข็ง ปอกเปลือก | 030617 | 5 ตัน",
    "ยางรถยนต์ เรเดียล 205/55R16 | 401110 | 400 เส้น",
    "น้ำตาลทรายขาว ICUMSA 45 | 170199 | 50 ตัน",
], title="ใบบรรจุหีบห่อ (Packing List)")

# Chinese customs declaration
create_invoice("customs_zh.png", [
    "冷冻虾仁 去头去壳 | 030617 | 10吨",
    "智能手机 安卓系统 6.7英寸 | 851712 | 5000台",
    "锂离子电池 电动汽车用 | 850760 | 200件",
    "热轧钢卷 宽600mm | 720839 | 100吨",
], title="报关单 (Customs Declaration)")

# Vietnamese invoice
create_invoice("invoice_vi.png", [
    "Tôm đông lạnh xuất khẩu | 030617 | 8 tấn",
    "Cà phê nhân xanh Robusta | 090111 | 20 tấn",
    "Gạo tẻ trắng 5% tấm | 100630 | 50 tấn",
], title="HÓA ĐƠN THƯƠNG MẠI")

print(f"\nAll sample documents saved to: {OUT_DIR}")
