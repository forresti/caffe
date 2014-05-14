/*
   Khalid Ashraf 2014
   Program to obtain the file path and MFCC start and 
   end location from the text file containing file names.
*/

#include <google/protobuf/text_format.h>
#include <glog/logging.h>
#include <leveldb/db.h>
#include "caffe/proto/caffe.pb.h"

#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <stdint.h>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <math.h>
using namespace std;

//class that defines the data structure returned by the 
class timeStamp{
    public:
    float startTime;
    float endTime;
    int phoneName;
    int numLines; 
};

vector<string> audioFileList(string fileName);

uint32_t swap_endian( uint32_t val )
{
    val = ((val << 8) & 0xFF00FF00 ) | ((val >> 8) & 0xFF00FF );
    return (val << 16) | (val >> 16);
}
uint16_t swap_endian16( uint16_t val )
{
    return (val << 8) | (val >> 8);
}

float reverseFloat( const float inFloat)
{
    float retVal;
    char *ftoC=(char*) & inFloat;
    char *retF=(char*) & retVal;

    retF[0]=ftoC[3];
    retF[1]=ftoC[2];
    retF[2]=ftoC[1];
    retF[3]=ftoC[0];
    return retVal;
}

std::vector< vector<float> > readMFCC(std::string fileName)
{
    uint32_t nSamples,samplePeriod;
    uint16_t sampSize,parmKind;
    float data;
    vector< vector<float> > features;

    std::ifstream mfccFile(fileName.c_str(), std::ios::in | std::ios::binary);


    if(mfccFile.is_open())
    {
        //read the header
        mfccFile.read((char*)(&nSamples), 4);
        nSamples = swap_endian(nSamples);
        mfccFile.read((char*)(&samplePeriod), 4);
        samplePeriod =swap_endian(samplePeriod);
        mfccFile.read((char*)(&sampSize), 2); //MFCC vector length in BYTES (e.g. 13-d mfcc float -> sampSize=56)
        sampSize =swap_endian16(sampSize);
        mfccFile.read((char*)(&parmKind), 2);
        parmKind =swap_endian16(parmKind);

        //go to the start of data
        mfccFile.seekg(12);

        std::cout<<"    numSamples: "<<nSamples<<"\n";
        std::cout<<"    sampSize: "<<sampSize<<"\n";

        int lineNum=0;
        //iterate through all the lines
        for(lineNum=0;lineNum<nSamples;++lineNum)
        {
            vector<float> currFeat(sampSize/4); //assume 13-d descriptors for now (could parse from file header later)
            features.push_back(currFeat); //append currFeat[lineIdx-1][:]
            //read the data for each line
            for(int i=0;i<(sampSize/4);i++)
            {
                //read the data(HTK saves files in Big Endian)
                mfccFile.read((char*)(&data), 4);
                //convert to little endian
                data=reverseFloat(data);
                //save it in a float array
                features[lineNum][i]=data;
                // pixels[i]=(uint8_t)round(data+100);
            }
        }
        // std::cout<<round(features[lineNum][0])<<"\n";
    }
    mfccFile.close();
    return features;
}

// read phone annotations from one file
std::vector<timeStamp> readLabel(std::string fileName)
{
    std::vector<timeStamp> timeStamps;
    int i;
    std::string outputS;
    //open file
    std::ifstream labelFile(fileName.c_str(), std::ios::in);

    if(labelFile.is_open()){
        i=0;
        while(!labelFile.eof())
        {
            //read one line at a time

            //lines look like this: startTime endTime phoneLabel

            if((i)%3==0) 
            {   
                timeStamps.resize( timeStamps.size() + 1 );
                std::getline(labelFile,outputS,' ');
                //std::cout<<"out"<<outputS<<'\n';
                timeStamps[i/3].startTime=::atof(outputS.c_str());
                //float dummy = ::atof(outputS.c_str());
                //std::cout<<timeStamps[i/3].startTime<<'\n';
                i++;
            }
            else if((i)%3==1)
            {
                std::getline(labelFile,outputS,' ');
                //std::cout<<"out"<<outputS<<'\n';
                timeStamps[i/3].endTime=::atof(outputS.c_str());
                //std::cout<<timeStamps[i/3].endTime<<'\n';
                i++;
            }

            else
            {
                //mapping from standard 39 TIMIT phone labels to numbers

                std::getline(labelFile,outputS,'\n');
                //std::cout<<"out"<<outputS<<'\n';

                if(outputS=="h#") timeStamps[i/3].phoneName=0;
                else if(outputS=="aa") timeStamps[i/3].phoneName=1;
                else if(outputS=="ae") timeStamps[i/3].phoneName=2;
                else if(outputS=="ah") timeStamps[i/3].phoneName=3;
                else if(outputS=="aw") timeStamps[i/3].phoneName=4;
                else if(outputS=="ay") timeStamps[i/3].phoneName=5;
                else if(outputS=="b") timeStamps[i/3].phoneName=6;
                else if(outputS=="ch") timeStamps[i/3].phoneName=7;
                else if(outputS=="d") timeStamps[i/3].phoneName=8;
                else if(outputS=="dh") timeStamps[i/3].phoneName=9;
                else if(outputS=="dx") timeStamps[i/3].phoneName=10;
                else if(outputS=="eh") timeStamps[i/3].phoneName=11;
                else if(outputS=="er") timeStamps[i/3].phoneName=12;
                else if(outputS=="ey") timeStamps[i/3].phoneName=13;
                else if(outputS=="f") timeStamps[i/3].phoneName=14;
                else if(outputS=="g") timeStamps[i/3].phoneName=15;
                else if(outputS=="hh") timeStamps[i/3].phoneName=16;
                else if(outputS=="ih") timeStamps[i/3].phoneName=17;
                else if(outputS=="iy") timeStamps[i/3].phoneName=18;
                else if(outputS=="jh") timeStamps[i/3].phoneName=19;
                else if(outputS=="k") timeStamps[i/3].phoneName=20;
                else if(outputS=="l") timeStamps[i/3].phoneName=21;
                else if(outputS=="m") timeStamps[i/3].phoneName=22;
                else if(outputS=="n") timeStamps[i/3].phoneName=23;
                else if(outputS=="ng") timeStamps[i/3].phoneName=24;
                else if(outputS=="ow") timeStamps[i/3].phoneName=25;
                else if(outputS=="oy") timeStamps[i/3].phoneName=26;
                else if(outputS=="p") timeStamps[i/3].phoneName=27;
                else if(outputS=="pau") timeStamps[i/3].phoneName=28;
                else if(outputS=="r") timeStamps[i/3].phoneName=29;
                else if(outputS=="s") timeStamps[i/3].phoneName=30;
                else if(outputS=="sh") timeStamps[i/3].phoneName=31;
                else if(outputS=="t") timeStamps[i/3].phoneName=32;
                else if(outputS=="th") timeStamps[i/3].phoneName=33;
                else if(outputS=="uh") timeStamps[i/3].phoneName=34;
                else if(outputS=="uw") timeStamps[i/3].phoneName=35;
                else if(outputS=="v") timeStamps[i/3].phoneName=36;
                else if(outputS=="w") timeStamps[i/3].phoneName=37;
                else if(outputS=="y") timeStamps[i/3].phoneName=38;
                else if(outputS=="z") timeStamps[i/3].phoneName=39;

                timeStamps[i/3].phoneName<<'\n';
                i++;
            }

        }//end of while
    } //end of file open
    labelFile.close();
    //labelSize=i/3;  //number of lines in the label file
    return timeStamps;
}

vector<int> timitToMfccLabel(int sizeMFCC, vector<timeStamp> ts)
{
    vector<int> mfccLabel;
    float labelTime[sizeMFCC],mfccTime[sizeMFCC];
    // float labelTime[219],mfccTime[219];


    std::cout<<"    sizeMfccTime: "<<sizeMFCC<<'\n';
    int j=0;  //index that iterates through the lines of the label file
    for (int i=0;i<sizeMFCC;i++)
    {
        mfccTime[i]=12.5+(i+1)*10; //real time in ms, i is the mfccID here
        //std::cout<<"mfccTime"<<mfccTime[i]<<'\n';

        labelTime[j]=ts[j].endTime*pow(10,-4); //real time in ms
        //std::cout<<"labelTime"<<labelTime[j]<<'\n';
        if(mfccTime[i]<labelTime[j]) {}
        else {j++;}

        mfccLabel.resize( mfccLabel.size() + 1 );
        mfccLabel[i]=ts[j].phoneName;
        //std::cout<<"mfccLabel"<<mfccLabel[i]<<'\n';
    }
    std::cout<<"    LabelSize: "<<j<<'\n';
    return mfccLabel;
}

void writeToLeveldb(char* db_filename,char* listFile,int windowSize)
{
    vector<string> fileList;
    std::vector<timeStamp> ts;
    std::vector< vector<float> > mfcc;
    std::vector<int> mfccLabel;
    int mfccTotal;     
    string mfccFileName,labelFileName;
    int sizeMFCC;
    char key[10];
    std::string value;
    // Open leveldb
    leveldb::DB* db;
    leveldb::Options options;
    options.create_if_missing = true;
    options.error_if_exists = true;
    leveldb::Status status = leveldb::DB::Open(
            options, db_filename, &db);
    CHECK(status.ok()) << "Failed to open leveldb " << db_filename
        << ". Is it already existing?";

    int numCoeff=23;
    //initialize the data structure
    caffe::Datum datum;
    datum.set_channels(1);
    datum.set_height(windowSize);//time axis
    datum.set_width(numCoeff); //mfcc axis, 23 for TNET features, will automate later

    int dbCount=0;
    mfccTotal=0;
    char* pixels=new char[numCoeff*windowSize];
    fileList=audioFileList(listFile);

    for(int i=0; i<fileList.size();i++)
    {
        /*
           labelFileName=fileList[i]+"lab";
           std::cout<<i<<labelFileName<<"\n";
         */

        mfccFileName=fileList[i]+"fea";
        labelFileName=fileList[i]+"lab";

        ts=readLabel(labelFileName);
        mfcc=readMFCC(mfccFileName);
        sizeMFCC=mfcc.size(); 
        mfccTotal+=sizeMFCC;
        mfccLabel=timitToMfccLabel(sizeMFCC,ts);

        cout << "    window labels for this file: ";
        for(int windowID=0;windowID<floor(sizeMFCC/windowSize);++windowID)
        { 
            for(int sampID=0;sampID<windowSize;sampID++)
            {
                for(int dataID=0;dataID<numCoeff;dataID++)
                {
                    pixels[sampID*numCoeff+dataID]=(char)(round(5*mfcc[(windowID*windowSize)+sampID][dataID])-0);                
                }
            }

            std::cout<< " " << (mfccLabel[windowID*windowSize+1]);

            //write current frame to leveldb
            datum.set_data((char*)pixels, numCoeff*windowSize);
            //  datum.set_data(pixels, numCoeff*windowSize);
            datum.set_label((char)mfccLabel[windowID*windowSize+1]);
            datum.SerializeToString(&value);
            sprintf(key, "%08d", dbCount);
            db->Put(leveldb::WriteOptions(), std::string(key), value);
            ++dbCount; 
        }
        cout << endl; //end of line for current window label list
        cout << endl; //done with printfs for this MFCC file
    }

    std::cout<<"numFiles: "<<fileList.size()<<"\n";
    std::cout<<"total mfcc: "<<mfccTotal<<"\n";
    std::cout<<"dbCount: "<<dbCount<<"\n";

}

vector<string> audioFileList(string fileName)
{
    int i;
    string output;
    vector<string> filePathList;
    //open the list file
    std::ifstream listFile(fileName.c_str(), std::ios::in);

    if(listFile.is_open())
    {
        //for (fileid=0;fileid<200;fileid++)
        while(!listFile.eof())
        {
            //read one line at a time
            listFile>>output;

            if(listFile.eof()==1)break;
            filePathList.resize( filePathList.size() + 1 );
            output.resize(output.size()-3);
            filePathList[i]=output;
            i++;
        }
    }
    listFile.close();
    return filePathList;
}

/*
usage: ./executable fileList leveldbName
 */

int main (int argc, char** argv)
{

    //int windowSize=3;
    int windowSize=23; //same size as TNet's postprocessed MFCC descriptors... square windows.
    /*    char* dbName=new char[200];
          dbName=argv[2];
          char* listFile=new char[200];
          listFile=argv[1];
     */
    writeToLeveldb(argv[2],argv[1],windowSize);

    return 0;
}


