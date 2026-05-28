from pathlib import Path
import sqlite3
from datetime import datetime, timedelta

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_docs"
RAW.mkdir(parents=True, exist_ok=True)
(ROOT / "outputs").mkdir(exist_ok=True)


def write_text_files():
    (RAW / "sop_fulfilment.txt").write_text("""
Judul: SOP Fulfilment Marketplace dan Internal
Departemen: Logistik
Versi: Approved 2026

Definisi SLA:
- Order marketplace mengikuti waktu kedaluwarsa platform.
- Order internal memiliki SLA H+1 pukul 23:59.
- Jika order internal masuk hari Sabtu, SLA dihitung ke hari Senin.

Eskalasi:
Jika terjadi kendala stok, SPV wajib menginformasikan ke SCM dan CS.
Jika keterlambatan disebabkan stok kosong, CS wajib memberi update ke customer.
""".strip(), encoding="utf-8")

    (RAW / "notulen_ops_mei_2026.txt").write_text("""
Notulen Operasional - 12 Mei 2026
Topik: Penurunan SLA Fulfilment Mei 2026

Keputusan:
1. Kendala utama terjadi pada channel TikTok dan Shopee.
2. Penyebab dominan adalah stok kosong dan antrean packing pada minggu kedua.
3. PIC perbaikan stok adalah Tim SCM.
4. PIC perbaikan jadwal packing adalah SPV Warehouse.
5. Evaluasi ulang dilakukan pada 20 Mei 2026.

Action item:
- SCM memperbarui safety stock untuk SKU fast moving.
- Warehouse menambah shift packing ketika order promo meningkat.
""".strip(), encoding="utf-8")

    (RAW / "glossary_kpi.md").write_text("""
# Glossary KPI

Revenue adalah total nilai penjualan sebelum dikurangi retur.
GMV adalah nilai transaksi kotor dari semua order.
Late order adalah order yang packed_at melebihi sla_deadline.
Late rate adalah persentase order terlambat dibanding total order.
Fulfilment adalah proses dari order masuk sampai paket siap dikirim.
""".strip(), encoding="utf-8")

    (RAW / "kebijakan_retur.txt").write_text("""
Kebijakan Retur Q2 2026

Retur produk rusak dapat diproses maksimal 7 hari sejak barang diterima customer.
Retur karena salah varian harus disertai video unboxing.
Retur tidak berlaku untuk produk clearance kecuali terjadi kerusakan pengiriman.
CS wajib membuat tiket retur dan menandai alasan retur pada sistem.
""".strip(), encoding="utf-8")


def write_pdf(filename: str, title: str, paragraphs: list[str]):
    path = RAW / filename
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]
    for text in paragraphs:
        story.append(Paragraph(text, styles["BodyText"]))
        story.append(Spacer(1, 8))
    doc = SimpleDocTemplate(str(path), pagesize=A4)
    doc.build(story)


def write_pdf_files():
    write_pdf(
        "sop_fulfilment_marketplace_internal.pdf",
        "SOP Fulfilment Marketplace dan Internal",
        [
            "Order marketplace mengikuti batas waktu platform. Jika platform menetapkan batas pengiriman hari yang sama, tim warehouse wajib memprioritaskan order tersebut.",
            "Order internal memiliki SLA H+1 pukul 23:59. Jika order internal masuk pada hari Sabtu, SLA dihitung ke hari Senin pukul 23:59.",
            "Jika terjadi kendala stok, SPV Warehouse wajib melakukan eskalasi kepada SCM dan Customer Service pada hari yang sama.",
        ],
    )
    write_pdf(
        "laporan_evaluasi_sla_mei_2026.pdf",
        "Laporan Evaluasi SLA Mei 2026",
        [
            "Pada Mei 2026, SLA fulfilment menurun terutama pada channel TikTok dan Shopee. Penurunan paling besar terjadi pada minggu kedua saat volume promo meningkat.",
            "Penyebab utama adalah stok kosong untuk SKU fast moving, antrean packing, dan keterlambatan handover ke kurir pada jam puncak.",
            "Rekomendasi: tambah safety stock, aktifkan shift tambahan saat promo, dan lakukan monitoring late order harian.",
        ],
    )
    write_pdf(
        "memo_kebijakan_retur_q2_2026.pdf",
        "Memo Kebijakan Retur Q2 2026",
        [
            "Retur karena produk rusak dapat diproses maksimal 7 hari setelah barang diterima customer.",
            "Retur salah varian wajib disertai bukti video unboxing dan foto label pengiriman.",
            "Produk clearance tidak dapat diretur kecuali terdapat kerusakan akibat pengiriman.",
        ],
    )
    write_pdf(
        "notulen_ops_12_mei_2026.pdf",
        "Notulen Operasional 12 Mei 2026",
        [
            "Keputusan rapat: channel TikTok menjadi prioritas perbaikan SLA karena late rate tertinggi pada minggu kedua Mei 2026.",
            "PIC perbaikan stok adalah Tim SCM. PIC perbaikan jadwal packing adalah SPV Warehouse.",
            "Deadline action item adalah 20 Mei 2026 dan hasilnya akan dievaluasi pada meeting operasional berikutnya.",
        ],
    )


def write_excel_file():
    channel_rows = [
        {"Periode": "Mei 2026", "Channel": "Shopee", "Revenue": 215000000, "Order": 1420, "Late_Order": 116, "Catatan": "Late naik saat promo minggu kedua."},
        {"Periode": "Mei 2026", "Channel": "TikTok", "Revenue": 188000000, "Order": 1325, "Late_Order": 151, "Catatan": "Kendala stok SKU fast moving."},
        {"Periode": "Mei 2026", "Channel": "Tokopedia", "Revenue": 98000000, "Order": 610, "Late_Order": 31, "Catatan": "Relatif stabil."},
        {"Periode": "Mei 2026", "Channel": "Website", "Revenue": 76500000, "Order": 410, "Late_Order": 22, "Catatan": "Mayoritas order internal."},
    ]
    glossary_rows = [
        {"KPI": "Revenue", "Definisi": "Total nilai penjualan sebelum retur."},
        {"KPI": "Late Rate", "Definisi": "Persentase order yang packed_at melebihi sla_deadline."},
        {"KPI": "Fulfilment", "Definisi": "Proses order masuk sampai paket siap dikirim."},
    ]
    action_rows = [
        {"Action": "Tambah safety stock", "PIC": "SCM", "Deadline": "2026-05-20", "Status": "Open"},
        {"Action": "Tambah shift packing promo", "PIC": "Warehouse", "Deadline": "2026-05-18", "Status": "In Progress"},
        {"Action": "Monitoring late order harian", "PIC": "BI Analyst", "Deadline": "2026-05-15", "Status": "Done"},
    ]
    path = RAW / "laporan_sales_mei_2026.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(channel_rows).to_excel(writer, sheet_name="Ringkasan_Channel", index=False)
        pd.DataFrame(glossary_rows).to_excel(writer, sheet_name="Glossary_KPI", index=False)
        pd.DataFrame(action_rows).to_excel(writer, sheet_name="Action_Plan", index=False)


def write_sqlite_db():
    db_path = ROOT / "data" / "retail_bi.db"
    db_path.parent.mkdir(exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    channels = ["Shopee", "TikTok", "Tokopedia", "Website"]
    skus = ["SKU-SERUM-A", "SKU-TONER-B", "SKU-CREAM-C", "SKU-SUNSCREEN-D"]
    rows = []
    base = datetime(2026, 5, 1, 9, 0, 0)
    for i in range(1, 241):
        channel = channels[i % len(channels)]
        sku = skus[i % len(skus)]
        order_dt = base + timedelta(hours=i * 3)
        sla_dt = order_dt + timedelta(days=1)
        late = (channel in ["TikTok", "Shopee"] and i % 5 == 0) or (i % 17 == 0)
        packed_dt = sla_dt + timedelta(hours=5 if late else -3)
        qty = (i % 4) + 1
        revenue = qty * (85000 + (i % 7) * 5000)
        rows.append((
            f"ORD-{i:04d}", order_dt.isoformat(sep=" "), channel, sku,
            qty, revenue, sla_dt.isoformat(sep=" "), packed_dt.isoformat(sep=" ")
        ))

    with sqlite3.connect(db_path) as con:
        con.execute("""
        CREATE TABLE orders (
            order_id TEXT PRIMARY KEY,
            order_date TEXT,
            channel TEXT,
            sku TEXT,
            qty INTEGER,
            revenue INTEGER,
            sla_deadline TEXT,
            packed_at TEXT
        )
        """)
        con.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        con.commit()


def main():
    write_text_files()
    write_pdf_files()
    write_excel_file()
    write_sqlite_db()
    print("Dataset berhasil dibuat.")
    print(f"Folder dokumen: {RAW}")
    print("Isi folder:")
    for path in sorted(RAW.glob("*")):
        print("-", path.name)


if __name__ == "__main__":
    main()
