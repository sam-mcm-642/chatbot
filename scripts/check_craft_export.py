"""
Quick check of Craft export structure
"""

from pathlib import Path
import re

def check_export():
    base_dir = Path("data/raw/Weekly targets")
    
    if not base_dir.exists():
        print(f"❌ Directory not found: {base_dir}")
        return
    
    md_files = list(base_dir.glob("**/*.md"))
    
    print("="*60)
    print("Craft Export Structure Check")
    print("="*60)
    print(f"\nTotal files: {len(md_files)}")
    
    # Analyze structure
    print("\nDirectory structure:")
    subdirs = set()
    for f in md_files:
        relative = f.relative_to(base_dir)
        parts = relative.parts[:-1]  # Exclude filename
        if parts:
            subdirs.add('/'.join(parts))
    
    for subdir in sorted(subdirs):
        count = len(list((base_dir / subdir).glob("*.md")))
        print(f"  {subdir}/ ({count} files)")
    
    # Check content
    print("\n" + "="*60)
    print("Sample Content Check")
    print("="*60)
    
    for f in sorted(md_files)[:3]:  # Check first 3
        print(f"\n📄 {f.name}")
        with open(f, 'r') as file:
            content = file.read()
            
            # Check for weekly targets format
            has_date_header = bool(re.search(r'#\s*::(\d{2}/\d{2}/\d{2}):\s*Weekly Targets::', content))
            task_count = len(re.findall(r'- \[[ x]\]', content))
            
            print(f"  Has date header: {'✓' if has_date_header else '✗'}")
            print(f"  Tasks found: {task_count}")
            print(f"  First line: {content.split(chr(10))[0][:60]}...")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    check_export()