# -----------------------
# Compiler/linker options
# -----------------------
CXX = g++
CXXFLAGS = -O3 -std=c++0x -c -Wall -Wno-sign-compare
LDFLAGS = -O3 -std=c++0x -Wno-sign-compare

# -----------
# Directories
# -----------

SRCDIR = src
BINDIR = bin
OBJDIR = build
DEPDIR = .deps
PROG = $(BINDIR)/despot

# -----
# Files
# -----

VPATH = $(shell find $(SRCDIR) -type d \( ! -name '.*' \))
SOURCES = $(shell find $(SRCDIR) -name '*.cpp')
OBJS = $(addprefix $(OBJDIR)/, $(patsubst %.cpp, %.o, $(notdir $(SOURCES))))
DEPS = $(addprefix $(DEPDIR)/, $(patsubst %.cpp, %.d, $(notdir $(SOURCES))))
BINS = $(PROG)
INCL = -I $(SRCDIR)

# -------
# Targets
# -------

.PHONY: all clean

all: DIR_TGTS $(DEPS) $(BINS)

DIR_TGTS:
	mkdir -p $(BINDIR) $(OBJDIR) $(DEPDIR)

$(PROG): $(OBJS)
	$(CXX) $(OBJS) $(LDFLAGS) $(INCL) -o $(PROG)

$(DEPDIR)/%.d: %.cpp
	@mkdir -p $(DEPDIR); \
	$(CXX) -MM $(CXXFLAGS) $(INCL) $< > $@; \
	sed -ie 's;\(.*\)\.o:;$(OBJDIR)/\1.o $(DEPDIR)/\1.d:;g' $@

-include $(DEPS)

$(OBJDIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCL) $< -o $@ 

clean:
	rm -rf $(OBJDIR) $(BINDIR) $(DEPDIR)
