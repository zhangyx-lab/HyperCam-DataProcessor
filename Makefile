ACQUIRE ?= $(shell realpath ../Acquire)
SUBDIR  := 5-Aligned
TARGETS := 23-03-31_69 23-03-31_77


default:
	@echo Available targets:
	@echo 1. clean   - Clean up calibration files
	@echo 2. DataSet - Gather all dataset files and create a zip


DataSet: dir $(patsubst %,$(ACQUIRE)/%, $(TARGETS))

dir:
	@rm -rf DataSet
	@mkdir -p DataSet/raw
	@mkdir -p DataSet/ref
	@mkdir -p DataSet/img


DataSet.zip: DataSet 
	@zip -r0 $@ DataSet/*
	@unzip -l $@


$(ACQUIRE_HOME)/%: dir
	$(info $@)
	$(eval PREFIX:=$(shell basename $@))
	$(eval LIST:=$(shell cat $@/$(SUBDIR)/list.txt | xargs))
	@for ID in $(LIST); do \
		ln -Tsf $@/$(SUBDIR)/raw/$$ID.npy DataSet/raw/$(PREFIX)_$$ID.npy; \
		ln -Tsf $@/$(SUBDIR)/ref/$$ID.npy DataSet/ref/$(PREFIX)_$$ID.npy; \
		ln -Tsf $@/$(SUBDIR)/$$ID.png DataSet/img/$(PREFIX)_$$ID.png; \
	done

clean:
	rm -rf $(shell python3 env.py | xargs)
	rm -rf data/runtime.ini

.PHONY: default dir DataSet DataSet.zip $(ACQUIRE_HOME)/% clean
