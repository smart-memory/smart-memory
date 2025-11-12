#!/bin/bash

success_count=0
failed_count=0

# First, remove existing PDFs to start fresh
rm -f *.pdf

echo "Converting all markdown files to PDF..."

# Find all markdown files recursively and convert them to PDF
find . -name "*.md" -type f | while read -r file; do
    # Create flat filename by replacing / with _
    flat_name=$(echo "$file" | sed 's|^\./||' | sed 's|/|_|g' | sed 's|\.md$|.pdf|')
    echo "Converting: $file -> $flat_name"
    
    # Use pandoc with pdflatex and sanitize Unicode characters
    if pandoc "$file" -o "$flat_name" --pdf-engine=pdflatex -V geometry:margin=1in 2>/dev/null || \
       pandoc "$file" -o "$flat_name" --pdf-engine=xelatex -V geometry:margin=1in 2>/dev/null; then
        echo "  ✓ Success"
        ((success_count++))
    else
        echo "  ✗ Failed - trying text cleanup"
        # Create a sanitized version
        temp_file=$(mktemp)
        # Remove problematic Unicode characters and replace with text equivalents
        sed 's/📚/[book]/g; s/🧠/[brain]/g; s/🚀/[rocket]/g; s/✅/[check]/g; s/❌/[x]/g; s/🌟/[star]/g; s/📊/[chart]/g; s/🔄/[cycle]/g; s/⚡/[lightning]/g; s/↔/↔/g; s/[├└─│]/|/g' "$file" > "$temp_file"
        
        if pandoc "$temp_file" -o "$flat_name" --pdf-engine=pdflatex -V geometry:margin=1in 2>/dev/null; then
            echo "  ✓ Success with cleanup"
            ((success_count++))
        else
            echo "  ✗ Failed even with cleanup"
            ((failed_count++))
        fi
        rm "$temp_file"
    fi
done

echo ""
echo "PDF conversion complete!"
echo "Successfully converted: $success_count files"  
echo "Failed conversions: $failed_count files"

# Show final count
echo ""
echo "Total PDF files created: $(ls *.pdf 2>/dev/null | wc -l)"