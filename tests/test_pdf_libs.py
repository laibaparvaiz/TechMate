"""
Quick standalone check: confirms `markdown` and `xhtml2pdf` are
installed correctly and can actually produce a real PDF together.

Run with: python test_pdf_libs.py
Delete this file afterward — it's just a one-time sanity check,
not part of the actual project.
"""

import sys

print("Checking imports...")

try:
    import markdown
    print(f"  markdown OK (version {markdown.__version__})")
except ImportError as e:
    print(f"  FAILED to import markdown: {e}")
    sys.exit(1)

try:
    from xhtml2pdf import pisa
    print("  xhtml2pdf OK (imported successfully)")
except ImportError as e:
    print(f"  FAILED to import xhtml2pdf: {e}")
    sys.exit(1)

print("\nGenerating a test PDF...")

sample_markdown = """
# Test Document

This confirms **markdown** and *xhtml2pdf* are working together.

## A list

- First item
- Second item
- Third item

## A code block

```
print("hello world")
```
"""

html_body = markdown.markdown(sample_markdown, extensions=["fenced_code"])

styled_html = f"""
<html>
<head>
<style>
    body {{ font-family: Helvetica, sans-serif; padding: 40px; }}
    h1 {{ color: #2c3e50; }}
    h2 {{ color: #34495e; }}
    code {{ background-color: #f4f4f4; padding: 2px 4px; }}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

output_path = "test_output_check.pdf"

with open(output_path, "wb") as output_file:
    result = pisa.CreatePDF(styled_html, dest=output_file)

if result.err:
    print(f"\nFAILED: xhtml2pdf reported {result.err} error(s) during generation.")
    sys.exit(1)

print(f"\nSuccess! Created '{output_path}'.")
print("Open it and confirm you see a styled document with a")
print("heading, a bulleted list, and a code block — not an error,")
print("not blank, not plain unstyled text.")