CXX = g++
CFLAGS = -g -fopenmp -O3 -fPIC -I/usr/local/include/ -I.. -I../eigen 
LDLIBS = -g -lgomp -ljpeg 
OBJS = JPEGPyramid.o JPEGImage.o Patchwork.o Rectangle.o PyramidStitcher.o test_stitch_pyramid.o
#OBJS = JPEGPyramid.o JPEGImage.o Patchwork.o Rectangle.o PyramidStitcher.o
all:	PyramidStitcher.so test_stitch_pyramid

# *** test suite ***

# OLD: link directly with .o files instead of linking with PyramidStitcher.so
test_stitch_pyramid: $(OBJS) 
	$(CXX) -o test_stitch_pyramid $(OBJS) $(LDLIBS)

#test_stitch_pyramid: test_stitch_pyramid.o
#	$(CXX) -o test_stitch_pyramid test_stitch_pyramid.o -L./libPyramidStitcher.so $(LDLIBS)
#	$(CXX) -L./lPyramidStitcher.so -o test_stitch_pyramid $(LDLIBS)

test_stitch_pyramid.o: ../test_stitch_pyramid.cpp ../PyramidStitcher.h ../JPEGPyramid.h ../JPEGImage.h ../Patchwork.h 
	$(CXX) $(CFLAGS) -c ../test_stitch_pyramid.cpp

# *** build deployable shared object binary ***
PyramidStitcher.so: $(OBJS)
	$(CXX) $(OBJS) -shared -o PyramidStitcher.so

PyramidStitcher.o: ../PyramidStitcher.cpp ../PyramidStitcher.h ../JPEGPyramid.h ../JPEGImage.h ../Patchwork.h
	$(CXX) $(CFLAGS) -c ../PyramidStitcher.cpp

JPEGPyramid.o:	../JPEGPyramid.cpp ../JPEGPyramid.h ../JPEGImage.h
	$(CXX) $(CFLAGS) -c ../JPEGPyramid.cpp 

JPEGImage.o: ../JPEGImage.cpp ../JPEGImage.h
	$(CXX) $(CFLAGS) -c ../JPEGImage.cpp 

Patchwork.o: ../Patchwork.cpp ../Patchwork.h ../JPEGPyramid.h ../Rectangle.h
	$(CXX) $(CFLAGS) -c ../Patchwork.cpp 

Rectangle.o: ../Rectangle.cpp ../Rectangle.h
	$(CXX) $(CFLAGS) -c ../Rectangle.cpp 

clean:
	rm -f *~ *.o *.mex* *.obj 

