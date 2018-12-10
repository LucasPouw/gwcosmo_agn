DOCS = gwcosmo_method.pdf

#FIGS = images/*.png

#BIBLIO = references.bib

BYPROD = *.aux *.log *.out *.dvi *.blg *.bbl

TEXFILES = gwcosmo_method.tex

all: $(DOCS)

default: all

Technical_Document.pdf: $(BIBLIO) $(TEXFILES)
	latexmk -pdf gwcosmo_method.tex

clean:
	rm -rf $(BYPROD) $(DOCS) arxiv.tar prl.tar prl gitID.txt
