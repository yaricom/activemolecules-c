#!/bin/sh


java -d64 -Xmx2G -jar tester.jar -s data/example_s.csv -data data/example_data.csv -exec "./activemoleculesC++"
