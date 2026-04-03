from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent 
INPUT_DIR = BASE_DIR / "extracted_texts"
OUTPUT_DIR = BASE_DIR / "clean_md"
OUTPUT_DIR.mkdir(exist_ok=True)

def remove_unwanted_headings(text: str) -> str:
    # Remove markdown H2 headings like:
    # ## Can ICE Enter a Home to Make an Arrest With Only an Administrative Warrant?
    text = re.sub(r'^\s*##\s+.*\n?', '', text, flags=re.MULTILINE)
    return text

def fix_line_breaks(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    buffer = []

    for line in lines:
        stripped = line.strip()

        # Preserve blank lines as paragraph boundaries
        if not stripped:
            if buffer:
                cleaned.append(" ".join(buffer))
                buffer = []
            cleaned.append("")
            continue

        # Preserve markdown headings other than the ones removed earlier
        if stripped.startswith("#"):
            if buffer:
                cleaned.append(" ".join(buffer))
                buffer = []
            cleaned.append(stripped)
            continue

        # Preserve list items
        if re.match(r"^[-*+]\s+", stripped) or re.match(r"^\d+\.\s+", stripped):
            if buffer:
                cleaned.append(" ".join(buffer))
                buffer = []
            cleaned.append(stripped)
            continue

        # Otherwise treat as wrapped paragraph text
        buffer.append(stripped)

    if buffer:
        cleaned.append(" ".join(buffer))

    # Collapse excessive blank lines
    result = "\n".join(cleaned)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip() + "\n"

def clean_markdown(text: str) -> str:
    text = remove_unwanted_headings(text)
    text = fix_line_breaks(text)
    return text

for path in INPUT_DIR.glob("*.md"):
    raw = path.read_text(encoding="utf-8")
    cleaned = clean_markdown(raw)
    out_path = OUTPUT_DIR / path.name
    out_path.write_text(cleaned, encoding="utf-8")
    print(f"Cleaned: {path.name}")