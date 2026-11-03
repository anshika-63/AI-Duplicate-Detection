import io
import zipfile
from pathlib import Path

sample_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>INVOICE #INV-2024-9021</w:t></w:r></w:p>
    <w:p><w:r><w:t>Date: November 2, 2026</w:t></w:r></w:p>
    <w:p><w:r><w:t>Bill To: Global Logistics Partners, 452 Innovation Blvd</w:t></w:r></w:p>
    <w:tbl>
      <w:tr>
        <w:tc><w:p><w:r><w:t>Item</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>Quantity</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>Price</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>Total</w:t></w:r></w:p></w:tc>
      </w:tr>
      <w:tr>
        <w:tc><w:p><w:r><w:t>Neural Network Consulting</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>1</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>$4500.00</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>$4500.00</w:t></w:r></w:p></w:tc>
      </w:tr>
    </w:tbl>
    <w:p><w:r><w:t>Subtotal: $4500.00</w:t></w:r></w:p>
    <w:p><w:r><w:t>Total Amount Due: $4500.00</w:t></w:r></w:p>
    <w:p><w:r><w:t>Payment Terms: Net 30 days via direct wire transfer</w:t></w:r></w:p>
  </w:body>
</w:document>"""

out_path = Path("cache/test_invoice.docx")
out_path.parent.mkdir(exist_ok=True)

with zipfile.ZipFile(out_path, "w") as z:
    z.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"></Types>')
    z.writestr("word/document.xml", sample_xml)

print(f"Created {out_path} ({out_path.stat().st_size} bytes)")
