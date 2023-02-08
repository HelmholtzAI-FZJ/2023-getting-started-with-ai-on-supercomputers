# The idea is to convert every Markdown file here into a HTML presentation using reveal.js

SOURCE_DOCS := $(wildcard *.md)

EXPORTED_DOCS=\
 $(SOURCE_DOCS:.md=.html)

RM=/bin/rm

PANDOC=/usr/local/bin/pandoc

PANDOC_OPTIONS=-t revealjs -s -V revealjs-url=https://unpkg.com/reveal.js --include-in-header=slides.css --embed-resources  -V hlss=zenburn -V theme=sky  # --no-highlight

%.html : %.md
	$(PANDOC) $(PANDOC_OPTIONS) -o $@ $<

.PHONY: all clean

all : $(EXPORTED_DOCS)

clean:
	- $(RM) $(EXPORTED_DOCS)