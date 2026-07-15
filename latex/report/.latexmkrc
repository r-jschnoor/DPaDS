# latexmk has built-in support for bibtex, but not for the glossaries
# package. This custom dependency (the standard recipe from the glossaries
# package's own docs, `texdoc glossaries` -> "Notes for latexmk users")
# teaches latexmk to regenerate the .gls file from .glo via makeindex
# whenever needed, so a plain `latexmk -pdf report.tex` builds the
# glossary automatically instead of requiring a manual `makeglossaries`
# step in between passes.
add_cus_dep('glo', 'gls', 0, 'makeglo2gls');
sub makeglo2gls {
    system("makeindex -s \"$_[0].ist\" -t \"$_[0].glg\" -o \"$_[0].gls\" \"$_[0].glo\"");
}
