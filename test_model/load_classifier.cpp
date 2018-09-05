#include "load_classifier.h"
#include "model.h" 
#include <zlib.h>

/*
model_gz[] was made from pretrained.xml using

gzip -c pretrained.xml >model.gz && \
xxd -i model.gz >model.h

*/
static inline std::string LoadForestData()
{

	std::string uncompressedData;

	const int bufferSize = 1024 * 1024;
	char* buffer = new char[bufferSize];

	z_stream strm = { 0 };
	strm.total_in = strm.avail_in = model_gz_len;
	strm.next_in = (Bytef*)model_gz;

	strm.zalloc = Z_NULL;
	strm.zfree = Z_NULL;
	strm.opaque = Z_NULL;

	int ret = inflateInit2(&strm, (15 + 32)); //15 window bits, and the +32 tells zlib to to detect if using gzip or zlib

	if (ret != Z_OK) {
		throw std::exception("Invalid forest");
	}

	do {
		strm.avail_out = bufferSize;
		strm.next_out = (Bytef *)buffer;
		ret = inflate(&strm, Z_NO_FLUSH);
		assert(ret != Z_STREAM_ERROR);  /* state not clobbered */
		switch (ret) {
		case Z_NEED_DICT:
		case Z_DATA_ERROR:
		case Z_MEM_ERROR:
			inflateEnd(&strm);
			throw std::exception("Invalid forest");
		}
		int have = bufferSize - strm.avail_out;

		uncompressedData.insert(uncompressedData.end(), &buffer[0], &buffer[have]);
	} while (ret != Z_STREAM_END);
	inflateEnd(&strm);

	delete[] buffer;

	return uncompressedData;
}


void ReadForest(CvRTrees& forest)
{
	std::string forestData = LoadForestData();

	CvFileStorage* fs = cvOpenFileStorage(forestData.c_str(), NULL, CV_STORAGE_READ | CV_STORAGE_MEMORY);

	CvFileNode* model_node = 0;
	CvFileNode* root = cvGetRootFileNode(fs);
	if (root->data.seq->total > 0) {
		model_node = (CvFileNode*)cvGetSeqElem(root->data.seq, 0);
	}
	forest.read(fs, model_node);
}