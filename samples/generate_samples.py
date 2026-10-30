"""Generate sample PDF and DOCX files for the Astra NX-200 demo knowledge base."""

from __future__ import annotations

from pathlib import Path
import docx


SAMPLES_DOCS_DIR = Path(__file__).resolve().parent / "docs"


def _escape_pdf_text(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )


def build_multipage_pdf(pages_lines: list[list[str]], output_path: Path) -> None:
    """Write a clean, standards-compliant multi-page PDF 1.4 readable by pypdf."""
    objects: list[bytes] = []

    def add_obj(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    # 1: Catalog placeholder, 2: Pages placeholder, 3: Helvetica Font, 4: Helvetica-Bold Font
    add_obj(b"")
    add_obj(b"")
    font_regular_id = add_obj(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    )
    font_bold_id = add_obj(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>"
    )

    page_obj_ids: list[int] = []

    for lines in pages_lines:
        stream_cmds: list[str] = ["BT"]
        y = 745
        for idx, line in enumerate(lines):
            if not line.strip():
                y -= 10
                continue
            is_heading = idx == 0 or line.startswith("## ")
            clean_line = line[3:] if line.startswith("## ") else line
            font_tag = "/F2" if is_heading else "/F1"
            font_size = 14 if idx == 0 else (11 if is_heading else 10)
            if is_heading and idx > 0:
                y -= 6
            escaped = _escape_pdf_text(clean_line)
            stream_cmds.append(f"{font_tag} {font_size} Tf")
            stream_cmds.append(f"1 0 0 1 50 {y} Tm ({escaped}) Tj")
            y -= 16 if not is_heading else 20
        stream_cmds.append("ET")
        stream_bytes = "\n".join(stream_cmds).encode("latin-1", errors="replace")

        content_id = add_obj(
            f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("ascii")
            + stream_bytes
            + b"\nendstream"
        )
        page_id = add_obj(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_regular_id} 0 R /F2 {font_bold_id} 0 R >> >> "
                f"/Contents {content_id} 0 R >>"
            ).encode("ascii")
        )
        page_obj_ids.append(page_id)

    kids_ref = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[1] = (
        f"<< /Type /Pages /Kids [{kids_ref}] /Count {len(page_obj_ids)} >>"
    ).encode("ascii")

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = [0]
    for obj_num, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{obj_num} 0 obj\n".encode("ascii"))
        out.extend(body)
        out.extend(b"\nendobj\n")

    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("ascii"))

    out.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n"
        ).encode("ascii")
    )
    output_path.write_bytes(bytes(out))


def generate_pdf_datasheet(target: Path) -> None:
    pages = [
        [
            "Astra NX-200 Industrial Edge Gateway - Hardware Datasheet (Page 1)",
            "Part Number: AST-NX200-R2   |   Hardware Revision: Rev 2.4",
            "",
            "## 1. Product Summary and Compute Architecture",
            "The Astra NX-200 (model AST-NX200-R2) is a rugged, fanless DIN-rail industrial edge",
            "gateway engineered for deterministic factory telemetry, protocol translation, and",
            "air-gapped plant-floor data aggregation.",
            "",
            "Processor: Quad-core ARM Cortex-A78AE industrial SoC operating at 2.2 GHz.",
            "System Memory: 8 GB LPDDR5 with inline Error-Correcting Code (ECC) at 4800 MT/s.",
            "Primary Storage: 64 GB industrial-grade pSLC eMMC 5.1 flash soldered on board.",
            "Expansion Storage: One internal M.2 2280 Key-M PCIe Gen3 x2 NVMe slot supporting",
            "up to 2 TB industrial NVMe SSD modules.",
            "Security Element: Dedicated discrete Infineon OPTIGA TPM 2.0 chip supporting hardware",
            "secure boot and cryptographic key storage.",
        ],
        [
            "Astra NX-200 Hardware Datasheet - Electrical & Environmental (Page 2)",
            "",
            "## 2. Electrical Power Specifications",
            "Input Voltage Range: Dual redundant terminal-block DC inputs rated 12 V to 48 V DC.",
            "Power Consumption: Typical nominal power draw is 18 W; peak load with PoE+ active is 34 W.",
            "Protection Circuitry: Integrated reverse-polarity protection up to -60 V DC and",
            "over-voltage crowbar clamping above 54 V DC.",
            "",
            "## 3. Mechanical and Environmental Ratings",
            "Operating Temperature: -30 C to +70 C continuous ambient with vertical DIN-rail airflow.",
            "Storage Temperature: -40 C to +85 C non-operating.",
            "Relative Humidity: 5% to 95% non-condensing.",
            "Enclosure Protection: IP40 black anodized extruded aluminum chassis, weight 1.15 kg.",
            "Vibration Resistance: 5 Grms random vibration (5 Hz to 500 Hz) per IEC 60068-2-64.",
            "Shock Resistance: 50 G half-sine 11 ms pulse per IEC 60068-2-27.",
        ],
        [
            "Astra NX-200 Hardware Datasheet - Network & Fieldbus I/O (Page 3)",
            "",
            "## 4. Ethernet and Industrial Fieldbus Interfaces",
            "Ethernet Port ETH0: 2.5GbE RJ45 uplink/management port (10/100/1000/2500BASE-T).",
            "Ethernet Port ETH1: 2.5GbE RJ45 downstream sensor port featuring IEEE 802.3at PoE+",
            "Power Sourcing Equipment (PSE) delivering up to 25.5 W (requires >= 24 V DC input).",
            "",
            "Serial Ports COM1 & COM2: Two software-selectable RS-485 / RS-232 DB9 ports featuring",
            "2.5 kVrms digital galvanic isolation and programmable 120-ohm line termination.",
            "Native protocol support for Modbus RTU master/slave up to 921.6 kbps.",
            "",
            "CAN Bus Interface: One isolated CAN-FD phoenix terminal port supporting 1 Mbps nominal",
            "arbitration bit rate and up to 5 Mbps data-phase bit rate (ISO 11898-1:2015).",
            "Digital I/O: 4 optically isolated digital inputs (24 V wet/dry) and 2 Form-A relay outputs.",
        ],
    ]
    build_multipage_pdf(pages, target)


def generate_docx_sop(target: Path) -> None:
    doc = docx.Document()
    doc.add_heading("Astra NX-200 Field Service & Preventive Maintenance SOP", level=0)
    doc.add_paragraph(
        "Standard Operating Procedure SOP-NX200-MAINT-03. Applicable to all AST-NX200-R2 units."
    )

    doc.add_heading("Preventive Maintenance Schedule", level=1)
    doc.add_paragraph(
        "Field technicians must execute the scheduled maintenance tasks listed below during planned plant maintenance windows:"
    )

    table = doc.add_table(rows=1, cols=3)
    hdr = table.rows[0].cells
    hdr[0].text = "Interval"
    hdr[1].text = "Component"
    hdr[2].text = "Required Action & Specification"

    rows = [
        (
            "Every 3 Months",
            "Heatsink Fins",
            "Clear accumulated dust using dry oil-free compressed air regulated below 2.0 bar (29 psi).",
        ),
        (
            "Every 12 Months",
            "DIN-Rail Latch & Ground Lug",
            "Verify M4 protective earth ground lug torque at 1.2 N-m and inspect spring latch engagement.",
        ),
        (
            "Every 36 Months",
            "RTC Backup Battery",
            "Replace internal wide-temperature coin cell with Panasonic CR2032W (-40 C to +85 C rated).",
        ),
    ]
    for interval, comp, action in rows:
        row_cells = table.add_row().cells
        row_cells[0].text = interval
        row_cells[1].text = comp
        row_cells[2].text = action

    doc.add_heading("Thermal Interface Pad Replacement", level=1)
    doc.add_paragraph(
        "Whenever the bottom service plate is opened after more than 24 months of field operation, "
        "the SoC thermal interface pad MUST be replaced rather than reused. Install genuine spare kit "
        "AST-TPAD-15, which contains a pre-cut 1.5 mm thick silicone-free thermal gap pad rated at "
        "6.0 W/mK thermal conductivity."
    )
    doc.add_paragraph(
        "Clean both the ARM SoC heat spreader and aluminum cold-plate pedestal with 99% isopropyl alcohol "
        "and lint-free wipes. Reinstall the four M3 service lid fasteners and tighten in a diagonal cross "
        "pattern to an exact torque of 0.65 N-m. Over-torquing beyond 0.80 N-m can crack the pad or strip threads."
    )

    doc.add_heading("Firmware Recovery Procedure", level=1)
    doc.add_paragraph(
        "If an interrupted power event corrupts the active OS partition and the gateway fails to boot, "
        "perform local offline recovery using the steps below:"
    )
    doc.add_paragraph(
        "1. Disconnect DC power from both terminal inputs. Press and hold the recessed RECOV pinhole button "
        "using a non-conductive nylon stylus while reapplying 24 V DC power. Keep holding RECOV for 7 seconds "
        "until the front-panel STAT LED flashes amber at 2 Hz."
    )
    doc.add_paragraph(
        "2. Connect a service laptop directly to port ETH1 and configure the laptop's Ethernet adapter with "
        "a static IP in the 192.168.200.0/24 subnet (for example 192.168.200.50). In recovery mode, the NX-200 "
        "listens on fallback static IP 192.168.200.1 via HTTPS port 8443 (https://192.168.200.1:8443)."
    )
    doc.add_paragraph(
        "3. Upload an Ed25519-signed firmware bundle (.astfw extension). Unsigned or modified images are "
        "rejected by the hardware TPM 2.0 bootloader."
    )

    doc.save(str(target))


def main() -> None:
    SAMPLES_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = SAMPLES_DOCS_DIR / "astra_nx200_datasheet.pdf"
    docx_path = SAMPLES_DOCS_DIR / "astra_nx200_maintenance_sop.docx"
    generate_pdf_datasheet(pdf_path)
    generate_docx_sop(docx_path)
    print(f"Generated: {pdf_path.name} ({pdf_path.stat().st_size} bytes)")
    print(f"Generated: {docx_path.name} ({docx_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
