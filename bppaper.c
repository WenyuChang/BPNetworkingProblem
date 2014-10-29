 
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define LINEAR 0
#define SIGMOID 1
unsigned char MASSINPUTS[8][8]= {
				{1,0,0,0,0,0,0,0},
				{0,1,0,0,0,0,0,0},
				{0,0,1,0,0,0,0,0},
				{0,0,0,1,0,0,0,0},
				{0,0,0,0,1,0,0,0},
				{0,0,0,0,0,1,0,0},
				{0,0,0,0,0,0,1,0},
				{0,0,0,0,0,0,0,1},
				};
unsigned char MASSTARGETS[8][8]={
				{1,0,0,0,0,0,0,0},
				{0,1,0,0,0,0,0,0},
				{0,0,1,0,0,0,0,0},
				{0,0,0,1,0,0,0,0},
				{0,0,0,0,1,0,0,0},
				{0,0,0,0,0,1,0,0},
				{0,0,0,0,0,0,1,0},
				{0,0,0,0,0,0,0,1},
				};
void readParamFile(char *filename, int *num_inputs, int *num_outputs,int *training_set_size, int *num_layers, int **arch, int **bias);
void printParams(int num_inputs, int num_outputs, int training_set_size,int num_layers, int *arch, int *bias);
void setupInputs(float ***unitinputs, int num_layers, int *arch, int *bias);
void zeroDeltaWeights(float ***deltaweights, int num_layers, int *arch, int *bias);
float runUnit(float *inputs, float *weights, int numInputs, int squasher);
void print2D(float **mat, int dim1, int dim2);
void printWeights(float ***weights, int numLayers, int *arch, int *bias);
float NthDerivative(int squasher, int deriv, float value);
float SigmoidAlpha = 1.0;
float RandomEqualREAL(float Low, float High)
{
  return ((float) rand() / RAND_MAX) * (High-Low) + Low;
}

int main(int argc, char **argv)
{
	FILE *file;
	int iterations, checkpoint, num_inputs, num_outputs, training_set_size,num_layers, i, j, k, h, p, x, lastlayer;
	/* arch: 各层结点数*/
	int *arch, *bias;
	unsigned char **squasher, **inputs, **targets;
	char infilename[80];
	float rho, floatin, TSS, thisError, out, sum, delta;
	float **outputs, **deltas;
	float ***weights, ***unitinputs, ***deltaweights, ***tempweights;
	long int totalbytes = 0;
	int checkpointOn = 0;
	int BPALG = 0;
	printf("SEQUENTIAL IMPLEMENTATION:\n");
	if(argc < 6) {
		fprintf(stderr, "Usage: %s [input parameter file] "
			"[# iterations] [learning rate (rho)] [BPALG (0=normal, 1=modified)]"
			"[checkpoint every # iterations] <weights file>\n", argv[0]);
		exit(1);
	}
	iterations = atoi(argv[2]);
	rho = atof(argv[3]);
	BPALG = atoi(argv[4]);
	checkpoint = atoi(argv[5]);
	printf("BPALG = %d\n", BPALG);
	readParamFile(argv[1], &num_inputs, &num_outputs, &training_set_size,&num_layers, &arch, &bias);
	printParams(num_inputs, num_outputs, training_set_size,num_layers, arch, bias);
	
	/*初始化输入输出层*/
	printf("Initializing input and output layers... ... \n");
	inputs = (unsigned char**)malloc(training_set_size*sizeof(unsigned char*));
	targets = (unsigned char**)malloc(training_set_size*sizeof(unsigned char*));
	for(i=0; i<training_set_size; i++) {
		totalbytes += 2*num_inputs*sizeof(unsigned char);
		inputs[i] = (unsigned char*)malloc(num_inputs*sizeof(unsigned char));
		targets[i] = (unsigned char*)malloc(num_outputs*sizeof(unsigned char));
		for(j=0; j<num_inputs; j++)
			inputs[i][j] = MASSINPUTS[i][j];
		for(j=0; j<num_outputs; j++)
			targets[i][j] = MASSTARGETS[i][j];
		printf("INPUTS:\t");
		for(j=0; j<num_inputs; j++)
			printf("\t%d", inputs[i][j]);
		printf("\nTARGETS:");
		for(j=0; j<num_outputs; j++)
			printf("\t%d", targets[i][j]);
		printf("\n");
		
		/*用于脸部识别功能的输入输出层初始化*/
		/* this code below is for face recognition */
		/*
		sprintf(infilename, "faces.NN/face_%d.NN", i+1);
		file = fopen(infilename, "r");
		if(!file) {
			perror("fopen");
			exit(1);
		}
		printf("\tOpened file: %s\n", infilename);
		fread(inputs[i], sizeof(unsigned char), num_inputs, file);
		fread(targets[i], sizeof(unsigned char), num_inputs, file);
		for(k=0; k<num_outputs; k++)
			printf("%d ", targets[i][k]);
		printf("\n");
		fclose(file);
		*/
	}
	/*初始化输入输出层结束*/
	
	/*初始化各层权值*/
	/*分配各层权值内存空间*/
	weights = (float***)malloc(num_layers*sizeof(float**));//当前权值
	tempweights = (float***)malloc(num_layers*sizeof(float**));//临时权值
	deltaweights = (float***)malloc(num_layers*sizeof(float**));//上次迭代权值
	unitinputs = (float***)malloc(num_layers*sizeof(float**));//每层的输入值
	for(i=0; i<num_layers; i++) {
		weights[i] = (float**)malloc(arch[i]*sizeof(float*));
		tempweights[i] = (float**)malloc(arch[i]*sizeof(float*));
		deltaweights[i] = (float**)malloc(arch[i]*sizeof(float*));
		unitinputs[i] = (float**)malloc(arch[i]*sizeof(float*));
		for(j=0; j<arch[i]; j++) {
			/* every layer but the first has weights of size of the previous
			 * layer */
			if(i != 0) {
				weights[i][j] = (float*)malloc((arch[i-1]+bias[i])*sizeof(float));
				tempweights[i][j] = (float*)malloc((arch[i-1]+bias[i])*sizeof(float));
				deltaweights[i][j] = (float*)malloc((arch[i-1]+bias[i])*sizeof(float));
				unitinputs[i][j] = (float*)malloc((arch[i-1]+bias[i])*sizeof(float));
			}
			/* for simplicity later, pretend input units have weights of 1 */
			else {
				weights[i][j] = (float*)malloc((1+bias[i])*sizeof(float));
				tempweights[i][j] = (float*)malloc((1+bias[i])*sizeof(float));
				deltaweights[i][j] = (float*)malloc((arch[i-1]+bias[i])*sizeof(float));
				unitinputs[i][j] = (float*)malloc((1+bias[i])*sizeof(float));
			}
		}
	}
	/*分配权值空间结束*/
	
	/*对各层当前权值赋值，采用两种赋值方式：1.随机数赋值；2.从文件读取*/
	/* if the user is not reading in weights from a file . . . */
	printf("Initializing weight parameters... ... \n");
	if(argc <= 6) {
		/* fill up the weight matrix with random gaussian values */
		//srand((unsigned int)time(NULL));
		for(i=0; i<num_layers; i++) {
			for(j=0; j<arch[i]; j++) {
				if(i != 0) 
					lastlayer = arch[i-1];
				else 
					lastlayer = 1;
				for(k=0; k<(lastlayer+bias[i]); k++) {
					if(i != 0) 
						weights[i][j][k] = RandomEqualREAL(0.0, 1.0);
					else 
						weights[i][j][k] = 1.0;
				}
			}
		}
	}
	/* they want to read weights in from a checkpoint file, do so now */
	else {
		printf("reading weights from '%s' file\n", argv[6]);
		file = fopen(argv[6], "r");
		if(!file) {
			perror("fopen");
			exit(1);
		}
		for(j=0; j<num_layers; j++) {
			for(k=0; k<arch[j]; k++) {
				if(j != 0) lastlayer = arch[j-1];
				else lastlayer = 0;
				for(h=0; h<(lastlayer+bias[j]); h++) {
					fscanf(file, "%f\t", &floatin);
					printf("floatin = %f\n", floatin);
					weights[j][k][h] = floatin;
				}
				if(lastlayer + bias[j] > 0)
					fscanf(file, "\n");
			}
		}
		fclose(file);
	}
	printf("Initializing weight parameters end. \n");
	/*赋值结束*/
	/*初始化各层权值结束*/
	
	/* alloc the matrices for deltas and outputs from each unit */
	outputs = (float**)malloc(num_layers*sizeof(float*));//各层输出向量
	deltas = (float**)malloc(num_layers*sizeof(float*));//？？
	squasher = (unsigned char**)malloc(num_layers*sizeof(unsigned char*));//用来判断采用计算输出向量的方式，
									      //参看代码最后NthDerivative方法。
	for(i=0; i<num_layers; i++) {
		outputs[i] = (float*)malloc(arch[i]*sizeof(float));
		deltas[i] = (float*)malloc(arch[i]*sizeof(float));
		squasher[i] = (unsigned char*)malloc(arch[i]*sizeof(unsigned char));
	}
	/* set up the squasher matrix */
	printf("seting up the squasher matrix... ... \n");
	for(i=0; i<num_layers; i++) {
		if(i == 0) 
			k = LINEAR;
		else 
			k = SIGMOID;
		for(j=0; j<arch[i]; j++) 
			squasher[i][j] = (unsigned char)k;
	}
	/*初始化各层单元输入值和上次迭代矩阵*/
	printf("Initializing unitinputs and deltaweights... ... \n");
	setupInputs(unitinputs, num_layers, arch, bias);
	zeroDeltaWeights(deltaweights, num_layers, arch, bias);
	
	/************************************************************************/
	/* BEGIN: main loop */
	/************************************************************************/
	printf("main loop starting... ... \n");
	for(i=0; i<iterations; i++) {
		printf("ITERATION %d\n", i);
		TSS = 0.0;
		for(k=1; k<num_layers; k++) {
			for(h=0; h<arch[k]; h++) {
				for(p=0; p<arch[k-1]+bias[k]; p++)
					deltaweights[k][h][p] = 0.0;
			}
		}
		for(j=0; j<training_set_size; j++) {
			thisError = 0.0;
			
			/* write the net inputs to the first layer unit inputs */
			/*将网络输入值写入第一层的单元输入*/
			for(k=0; k<arch[0]; k++)
				unitinputs[0][k][0] = inputs[j][k];
				
			/*计算各层的输入值和输出值*/
			for(k=0; k<num_layers; k++) {
				for(h=0; h<arch[k]; h++) {
					if(k != 0) 
						lastlayer = arch[k-1];
					/* if it's the input layer then there is always one input */
					else 
						lastlayer = 1;
					/*计算输出值*/
					out = runUnit(unitinputs[k][h], weights[k][h],lastlayer+bias[k], squasher[k][h]);
					/* record the output of this unit */
					outputs[k][h] = out;
					/* put the results into unitinputs matrix for next layer */
					/*计算输入值*/
					if(k+1 != num_layers) {
						/* except for the last layer */
						for(x=0; x<arch[k+1]; x++)
							unitinputs[k+1][x][h] = out;
					}
				}
			}
			
			/*计算误差*/
			/* calculate error */
			for(k=0; k<arch[num_layers-1]; k++)
				thisError += pow((((float)targets[j][k])-outputs[num_layers-1][k]),2);
			thisError *= 0.5;
			if(i == iterations - 1) 
				printf("Error #%d = %f\n", j, thisError);
			/* add this into the TSS for this iteration */
			TSS += thisError;
			
			/*以下的CHECKPOITING代码完成断点检查的功能*/
			/******************************************************************/
			/* CHECKPOINTING: implement that here
			/******************************************************************/
			if(checkpoint > 0 && i != 0 && (i % checkpoint == 0 || i == iterations-1)) {
				if(checkpointOn == 0) {
					file = fopen("checkpoint", "w");
					if(!file) {
						perror("fopen");
						exit(1);
					}
					for(x=0; x<num_layers; x++) {
						for(k=0; k<arch[x]; k++) {
							if(x != 0) 
								lastlayer = arch[x-1];
							else 
								lastlayer = 0;
							for(h=0; h<(lastlayer+bias[x]); h++) {
								fprintf(file,"%f\t", weights[x][k][h]);
							}
							if(lastlayer + bias[x] > 0)
								fprintf(file,"\n");
						}
					}
					checkpointOn = 1;
				}
				for(k=0; k<arch[num_layers-1]; k++) {
					fprintf(file,"%3.5f", inputs[j][k]);
					fprintf(file,"\t%3.5f", targets[j][k]);
					fprintf(file,"\t%3.5f\n", outputs[num_layers-1][k]);
				}
				if(j == iterations-1) {  //j == iterations-1??这里的j是否应该是i
					fclose(file);
					checkpointOn = 0;
				}
			}
			/******************************************************************/
			/* CHECKPOINTING: end
			/******************************************************************/
			if(i == iterations-1) {
				for(k=0; k<arch[num_layers-1]; k++) {
					printf("\t%d", targets[j][k]);
					printf("\t%3.5f\n", outputs[num_layers-1][k]);
				}
			}
			
			/*误差反传开始*/
			/******************************************************************/
			/* GDR: backprop starts here
			/******************************************************************/
			for(k=num_layers-1; k>=1; k--) {
				for(h=0; h<arch[k]; h++) {
					delta = 0.0;
					/* if last layer, then delta calculation is different */
					if(k == num_layers - 1) {
						if(BPALG == 0) {
						delta = (((float)targets[j][h]) - outputs[k][h]) * NthDerivative(squasher[k][h],1,outputs[k][h]);
						}
						else if(BPALG == 1) {
							delta = 4.0*pow(((float)targets[j][h])-outputs[k][h],3.0)
							*exp((((float)targets[j][h])-outputs[k][h])*
							(((float)targets[j][h])-outputs[k][h]))*NthDerivative(squasher[k][h],1,outputs[k][h]);
						}
					} else {
						sum = 0.0;
						/* SUM(delta_n * Wnj) */
						for(p=0; p<arch[k+1]; p++)
							sum += deltas[k+1][p] * weights[k+1][p][h];
						delta = NthDerivative(squasher[k][h],1,outputs[k][h]) * sum;
					}
					deltas[k][h] = delta;
					/* now calculate deltaweights */
					for(p=0; p<arch[k-1]+bias[k]; p++) {
						/* equation 6.32 */
						/* we're doing epoch only, if doing 'by pattern'
						 * then change this code below */
						deltaweights[k][h][p] += rho * delta *unitinputs[k][h][p] + deltaweights[k][h][p];
						//printf("dweight[%d][%d][%d] = %f\n", k, h, p, deltaweights[k][h][p]);
					}
				}
			}
			/******************************************************************/
			/* GDR: backprop end
			/******************************************************************/
		}
		/* apply the delta weights now */
		for(k=1; k<num_layers; k++) {
			for(h=0; h<arch[k]; h++) {
				for(p=0; p<arch[k-1]+bias[k]; p++)
					weights[k][h][p] += deltaweights[k][h][p];
			}
		}
		//printf("weights[0][0][0] = %f, weights[1][0][0] = %f\n",
		// weights[0][0][0], weights[1][0][0]);
		if(i % 1000 == 0)
			printf("%d\t%f\n", i, TSS*2);
	}
	/************************************************************************/
	/* END: main loop */
	/************************************************************************/
	/* we're not going to free all the memory here b/c looping and freeing all
	 * the memory we allocated would take longer than letting the OS free the
	 * entire section of memory when the program ends. If this is going to be
	 * used for something else (i.e: the program doesn't end here) then all this
	 * memory must be freed. */
	return 1;
}

void readParamFile(char *filename, int *num_inputs, int *num_outputs,int *training_set_size, int *num_layers, int **arch, int **bias){
	FILE *infile;
	int i;
	int junk;
	printf("Reading parameters file... ... \n");
	infile = fopen(filename, "r");
	if(!infile) {
		printf("Reading parameters file failed \n");
		perror("fopen");
		exit(1);
	}
	fscanf(infile, "num_inputs: %d\n", num_inputs);
	fscanf(infile, "num_outputs: %d\n", num_outputs);
	fscanf(infile, "training_set_size: %d\n", training_set_size);
	fscanf(infile, "num_layers: %d\n", num_layers);
	/* malloc room for the dimensions of the layers and read them in */
	*arch = (int*)malloc(*num_layers * sizeof(int));
	fscanf(infile, "layers: ");
	for(i=0; i<*num_layers; i++) 
		fscanf(infile, "%d\t", &((*arch)[i]));
	fscanf(infile, "\n");
	/* read in booleans for whether each layer has a bias input */
	fscanf(infile, "layer_biases: ");
	*bias = (int*)malloc(*num_layers * sizeof(int));
	for(i=0; i<*num_layers; i++) 
		fscanf(infile, "%d\t", &((*bias)[i]));
	fclose(infile);
	printf("Reading parameters file end\n");
}

void printParams(int num_inputs, int num_outputs, int training_set_size,int num_layers, int *arch, int *bias){
	int i;
	printf("num_inputs: %d\n", num_inputs);
	printf("num_outputs: %d\n", num_outputs);
	printf("training_set_size: %d\n", training_set_size);
	printf("num_layers: %d\n", num_layers);
	printf("layers: ");
	for(i=0; i<num_layers; i++) 
		printf("%d ", arch[i]);
	printf("\n");
	printf("layer_biases: ");
	for(i=0; i<num_layers; i++) 
		printf("%d ", bias[i]);
	printf("\n");
}

void setupInputs(float ***unitinputs, int num_layers, int *arch, int *bias){
	int i, j, k, lastlayer;
	printf("Initializing unitinputs... ... \n");
	for(i=0; i<num_layers; i++) {
		for(j=0; j<arch[i]; j++) {
			if(i != 0) 
				lastlayer = arch[i-1];
			else 
				lastlayer = 1;
			for(k=0; k<(lastlayer+bias[i]); k++)
				unitinputs[i][j][k] = 1.0;
		}
	}
	printf("Initializing unitinputs end \n");
}

void zeroDeltaWeights(float ***deltaweights, int num_layers, int *arch,int *bias){
	int i, j, k, lastlayer;
	printf("Initializing deltaweights... ... \n");
	for(i=0; i<num_layers; i++) {
		for(j=0; j<arch[i]; j++) {
			if(i != 0) 
				lastlayer = arch[i-1];
			else 
				lastlayer = 1;
			for(k=0; k<(lastlayer+bias[i]); k++)
				deltaweights[i][j][k] = 0.0;
		}
	}
	printf("Initializing deltaweights end \n");
}

float runUnit(float *inputs, float *weights, int numInputs, int squasher){
	int i;
	float net = 0.0;
	float out;
	for(i=0; i<numInputs; i++)
		net += (inputs[i] * weights[i]);
	return NthDerivative(squasher, 0, net);
}

void print2D(float **mat, int dim1, int dim2) {
	int i, j;
	for(i=0; i<dim1; i++)
		for(j=0; j<dim2; j++)
			printf("\tmat[%d][%d] = %f\n", i, j, mat[i][j]);
}

void printWeights(float ***weights, int numLayers, int *arch, int *bias){
	int i, j, k, lastlayer;
	for(i=0; i<numLayers; i++) {
		printf("Layer[%d]\n", i);
		for(j=0; j<arch[i]; j++) {
			printf("\tU[%d]\n", j);
			if(i != 0) lastlayer = arch[i-1];
			else lastlayer = 0;
			for(k=0; k<(lastlayer+bias[i]); k++) {
				printf(" w[%d][%d][%d] = %f", i, j, k, weights[i][j][k]);
			}
			printf("\n");
		}
	}
}

float NthDerivative(int squasher, int deriv, float value){
	float out;
	switch(squasher) {
		case SIGMOID:
			switch(deriv) {
				case 0:
					out = (1.0/(1.0+exp(-(SigmoidAlpha*value))));
					break;
				case 1:
					out = value*(1.0-value);
					break;
				default:
					printf("DERIV. NOT IMPLEMENTED.\n");
					exit(1);
			}
			break;
		case LINEAR:
			switch(deriv) {
				case 0:
					out = value;
					break;
				default:
					printf("DERIV. NOT IMPLEMENTED.\n");
					exit(1);
			}
			break;
		default:
			printf("INVALID SQUASHER\n");
			exit(1);
	}
	return out;
}
