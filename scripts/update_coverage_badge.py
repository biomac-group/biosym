#!/usr/bin/env python3
"""
Script to automatically update the coverage badge in README.md
This script parses the coverage.xml file and updates the coverage percentage in the README badge.
"""

import xml.etree.ElementTree as ET
import re
import sys
from pathlib import Path


def get_coverage_percentage(coverage_xml_path: Path) -> int:
    """Parse coverage.xml and extract the overall coverage percentage."""
    try:
        tree = ET.parse(coverage_xml_path)
        root = tree.getroot()
        
        # Find the coverage element and get the line-rate attribute
        coverage_element = root.find('.')
        if coverage_element is not None and 'line-rate' in coverage_element.attrib:
            line_rate = float(coverage_element.attrib['line-rate'])
            return int(round(line_rate * 100))
        
        # Alternative: look for summary statistics
        for elem in root.iter():
            if elem.tag == 'coverage' and 'line-rate' in elem.attrib:
                line_rate = float(elem.attrib['line-rate'])
                return int(round(line_rate * 100))
                
        print("Could not find coverage percentage in XML file")
        return 0
        
    except (ET.ParseError, FileNotFoundError, ValueError) as e:
        print(f"Error parsing coverage XML: {e}")
        return 0


def update_readme_badge(readme_path: Path, coverage_percentage: int) -> bool:
    """Update the coverage badge in README.md with the new percentage."""
    try:
        content = readme_path.read_text()
        
        # Pattern to match the coverage badge
        badge_pattern = r'(\[!\[Coverage\]\(https://img\.shields\.io/badge/coverage-)\d+(%25-[^)]+\))'
        
        # Determine color based on coverage percentage
        if coverage_percentage >= 90:
            color = "brightgreen"
        elif coverage_percentage >= 80:
            color = "green"
        elif coverage_percentage >= 70:
            color = "yellow"
        elif coverage_percentage >= 60:
            color = "orange"
        else:
            color = "red"
        
        new_badge = f"\\g<1>{coverage_percentage}%25-{color})"
        
        # Replace the badge
        new_content = re.sub(badge_pattern, new_badge, content)
        
        if new_content != content:
            readme_path.write_text(new_content)
            print(f"Updated coverage badge to {coverage_percentage}%")
            return True
        else:
            print("No coverage badge found to update")
            return False
            
    except Exception as e:
        print(f"Error updating README: {e}")
        return False


def main():
    """Main function to update coverage badge."""
    project_root = Path(__file__).parent.parent
    coverage_xml = project_root / "coverage.xml"
    readme_path = project_root / "README.md"
    
    if not coverage_xml.exists():
        print("coverage.xml not found. Run tests with coverage first:")
        print("python -m pytest tests/ --cov=biosym --cov-report=xml")
        sys.exit(1)
    
    if not readme_path.exists():
        print("README.md not found")
        sys.exit(1)
    
    coverage_percentage = get_coverage_percentage(coverage_xml)
    
    if coverage_percentage > 0:
        success = update_readme_badge(readme_path, coverage_percentage)
        if success:
            print(f"Successfully updated coverage badge to {coverage_percentage}%")
        else:
            print("Failed to update coverage badge")
            sys.exit(1)
    else:
        print("Could not determine coverage percentage")
        sys.exit(1)


if __name__ == "__main__":
    main()
