#include<iostream>
#include<fstream>
#include<cstring>
#include<cstdlib>
#include<sstream>
#include<cmath>
#include<vector>
#include<queue>
using namespace std;

const int maxTrainSize = 23786 + 100;
const int maxTestSize = 15858 + 100;
const int maxValidSize = 4000;
const int maxFeatureSize = 65;
const double doubleINF = 1e300;
const int intINF = 1000000007;
vector<double> vec_train[maxTrainSize], vec_valid[maxValidSize], vec_test[maxTestSize]; // store the feature
int label_train[maxTrainSize], label_valid[maxValidSize], label_test[maxTestSize]; // store the label

void readFile(vector<double>* vec, int* label, int& sz, const char* fileName, bool isTrain) {
	cout << "Begin readFile \"" << fileName << "\"" <<  endl;
	sz = 0;
	ifstream in(fileName);
	if(!in.is_open()) {
		cout << "Error opening file: " << fileName << endl;
		exit(1);
	}
	string token, Line;
	getline(in,Line);
	while(!in.eof()) {
		getline(in,Line);
		if(Line.size() == 0) break;
		stringstream ss(Line);
		vector<int> tmp;
		vec[sz].push_back(1.0);
		while(getline(ss,token,',')) {
			tmp.push_back(atoi(token.c_str()));
			vec[sz].push_back(atof(token.c_str()));	
		}
		label[sz] = tmp.back();
		vec[sz++].pop_back();
	}
	in.close();
	cout << "Finish readFile \"" << fileName << "\"" <<  endl;
}

double maxFeatureValue[maxFeatureSize], minFeatureValue[maxFeatureSize];
double w[maxFeatureSize]; // the weight vector
int featureSize, trainSize, validSize, testSize;
void init() {
	cout << "Begin init." << endl;
	for(int i = 0; i < featureSize; i++) {
		maxFeatureValue[i] = -doubleINF; minFeatureValue[i] = doubleINF;
		for(int j = 0; j < trainSize; j++) {
			maxFeatureValue[i] = max(maxFeatureValue[i],vec_train[j][i]);
			minFeatureValue[i] = min(minFeatureValue[i],vec_train[j][i]);
		}
		// normalization
		if(fabs(maxFeatureValue[i]-minFeatureValue[i]) < 1e-8) continue; // continue when the denominator is zero
		for(int j = 0; j < trainSize; j++) vec_train[j][i] = (vec_train[j][i]-minFeatureValue[i]) / (maxFeatureValue[i] - minFeatureValue[i]);
		for(int j = 0; j < validSize; j++) vec_valid[j][i] = (vec_valid[j][i]-minFeatureValue[i]) / (maxFeatureValue[i] - minFeatureValue[i]);
		for(int j = 0; j < testSize; j++) vec_test[j][i] = (vec_test[j][i]-minFeatureValue[i]) / (maxFeatureValue[i] - minFeatureValue[i]);
	}
	memset(w,0,sizeof(w));
	cout << "Finish init." << endl;
}

void training(int iterations) {
	double s[maxTrainSize], grad[maxFeatureSize];
	for(int step = 0; step < iterations; step++) {
		memset(s,0,sizeof(s));		
		// calculate the dot product
		for(int i = 0; i < trainSize; i++) {
			s[i] = 0;
			for(int j = 0; j < featureSize; j++) s[i] += (w[j] * vec_train[i][j]);
		}
		// calculate the gradient
		for(int f = 0; f < featureSize; f++) {
			grad[f] = 0.0;
			for(int i = 0; i < trainSize; i++) {
				grad[f] += (1.0/(1.0+exp(-s[i])) - label_train[i]) * vec_train[i][f];
			}
		}
		// updata the weight vector
		for(int f = 0; f < featureSize; f++) {
			w[f] = w[f] - 1.0/(10.0*abs(w[f])+300.0*(step+1.0)) * grad[f];
		}
	}
	double sum = 0.0;
	for(int f = 0; f < featureSize; f++) {
		cout << "before: " << w[f] << endl;
		sum += w[f];
	}
}

// use LR to classify test[_id]
int LR(vector<double>* _vec, int _id) {
	double s = 0.0;
	for(int f = 0; f < featureSize; f++) s += w[f] * _vec[_id][f];
	double h = 1.0 / (1.0 + exp(-s));
	return h > (0.4-1e-6)? 1: 0;
}

void validating(vector<double>* _vec, int _label[],  int _size, double &acc, double &rec, double &prc, double &F1) {
	cout << "Begin validating" << endl;
	int TP = 0, FN = 0, FP = 0, TN = 0;
	for(int i = 0; i < _size; i++) {
		int calc = LR(_vec,i);
		int ans = _label[i];
		if(ans == 1 && calc == 1) TP++;
		if(ans == 1 && calc == 0) FN++;
		if(ans == 0 && calc == 1) FP++;
		if(ans == 0 && calc == 0) TN++; 
	}
	acc = 1.0 * (TP + TN) / (TP + FP + TN + FN);
	rec = 1.0 * TP / (TP + FN);
	prc = 1.0 * TP / (TP + FP);
	F1 = 2.0 * (prc * rec) / (prc + rec);
	cout << TP << " " << FN << " " << FP << " " << TN << endl;
	cout << "Finish validating" << endl;
}

void testing() {
	cout << "Begin testing" << endl;
	char fileName[] = "112_1_v20.txt";
	ofstream out(fileName);
	if(!out.is_open()) {
		cout << "Error in opening file \"" << fileName << "\"" << endl;
		exit(1);
	}
	
	for(int i = 0; i < testSize; i++) {
		out << LR(vec_test,i) << endl;
	}
	
	out.close();
	cout << "Finish testing" << endl;
}

int main() {
	string fileName = "train.csv";
	readFile(vec_train,label_train,trainSize,fileName.c_str(),1);
	fileName = "validation_1231_v10.csv";
	readFile(vec_valid,label_valid,validSize,fileName.c_str(),0);
	fileName = "test.csv";
	readFile(vec_test,label_test,testSize,fileName.c_str(),0);
	featureSize = vec_train[0].size();
	cout << trainSize << endl;
	cout << validSize << endl;
	cout << featureSize << endl;
	init();
	training(3000);
	for(int i = 0; i < featureSize; i++) cout << w[i] << endl;
	double acc, rec, prc, F1;
	validating(vec_valid,label_valid,validSize,acc,rec,prc,F1);
	cout << "Accuracy: " << acc << endl;
	cout << "Recall: " << rec << endl;
	cout << "Precision: " << prc << endl;
	cout << "F1: " << F1 << endl;
	testing();
	system("pause");
	return 0;
}
