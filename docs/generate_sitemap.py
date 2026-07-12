# Authors:
#
#       Bill Sousa
#
# License: BSD 3 clause
#


import os
from pathlib import Path

# RTD sets this environment variable during build
output_dir = Path(os.environ["READTHEDOCS_OUTPUT"]) / "html"

base_url = "https://pybear.readthedocs.io/en/stable/"

# Pages to exclude from sitemap
exclude = {"genindex.html", "search.html", "py-modindex.html"}

urls = []
for html_file in sorted(output_dir.rglob("*.html")):
    # Get path relative to output_dir
    relative = html_file.relative_to(output_dir)
    parts = relative.parts

    # Strip leading 'en/' if present
    if parts[0] == "en":
        relative = Path(*parts[1:])

    # Skip excluded pages and _static directory
    if relative.name in exclude or "_static" in relative.parts:
        continue

    urls.append(f"{base_url}{relative}")

# Write sitemap.xml
sitemap_path = output_dir / "sitemap.xml"
with open(sitemap_path, "w", encoding="utf-8") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n')
    for url in urls:
        f.write(f"  <url><loc>{url}</loc></url>\n")
    f.write("</urlset>\n")

print(f"sitemap.xml written with {len(urls)} URLs to {sitemap_path}")
