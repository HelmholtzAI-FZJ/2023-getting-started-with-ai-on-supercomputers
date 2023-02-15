# The idea is to convert every Markdown file here into a HTML presentation using reveal.js

SOURCE_DOCS := $(wildcard *.md)

EXPORTED_DOCS=\
  $(addprefix public/,$(SOURCE_DOCS:.md=.html))

RM=/bin/rm

PANDOC=/usr/local/bin/pandoc

PANDOC_OPTIONS=-t revealjs -s \
	-V revealjs-url=https://unpkg.com/reveal.js \
	--include-in-header=slides.css \
	-V hlss=zenburn \
	-V theme=sky \
	-V transition=fade \
	# --embed-resources \ 
	# -A footer.html # The footer is just too big

public/%.html : %.md *.css
	$(PANDOC) $(PANDOC_OPTIONS) -o $@ $<

.PHONY: all clean

all : $(EXPORTED_DOCS)

clean:
	- $(RM) $(EXPORTED_DOCS)