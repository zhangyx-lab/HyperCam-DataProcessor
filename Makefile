clean:
	rm -rf $(shell python3 env.py | xargs)
	rm -rf data/runtime.ini

.PHONY: clean
