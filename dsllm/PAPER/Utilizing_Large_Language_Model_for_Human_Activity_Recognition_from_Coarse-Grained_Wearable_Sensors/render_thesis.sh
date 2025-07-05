#!/bin/bash
# set -e

# Change to the thesis directory
cd "$(dirname "$0")"

MAIN_TEX="ncku_thesis.tex"
PDF_NAME="ncku_thesis.pdf"

# Clean up auxiliary files (optional)
echo "Cleaning up old build files..."
rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fdb_latexmk *.fls *.run.xml *.synctex.gz

# First LaTeX run (generates .aux file)
echo "Running first LaTeX pass..."
pdflatex -interaction=nonstopmode "$MAIN_TEX"

# Run BibTeX for bibliography
echo "Running BibTeX..."
bibtex "${MAIN_TEX%.tex}"

# Second LaTeX run (resolves citations)
echo "Running second LaTeX pass..."
pdflatex -interaction=nonstopmode "$MAIN_TEX"

# Final LaTeX run (finalizes document)
echo "Running final LaTeX pass..."
pdflatex -interaction=nonstopmode "$MAIN_TEX"

# Open the resulting PDF (optional)
# echo "Opening the PDF..."
# xdg-open "$PDF_NAME" &

echo "Build complete!" 