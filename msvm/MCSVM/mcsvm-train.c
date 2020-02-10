/* $Author: kobics $ */
/* $Date: 2003/10/01 11:49:09 $ */
/* $Source: /cs/phd/kobics/.CVSROOT/code/multiClass/mcsvm-train.c,v $ */
/* $Name:  $ */
/* $Locker:  $ */
/* $Revision: 6.7.2.1 $ */
/* $State: Exp $ */



#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include "spoc.h"
#include "utilities.h"
#include "mucutils.h"

#define STRLEN 256

static MCScaleDef mc_scaledef;
static MCDataDef mc_datadef;
static SPOCParamDef spoc_pd;

static char *variables[] = {
  "m", "k", "l",
  "beta", "cache_size",
  "kernel_type", "polynom_degree", "polynom_a0", "exponent_sigma",
  "epsilon", "epsilon0", "delta", "redopt_type",
  "scale_file",
  "ENDOFTABLE"
};

static char train_data_name[STRLEN];
static char classifier_name[STRLEN];
static char scale_file[STRLEN];

void initialize();
long  read_parameters_from_file(char *file_name);
long  read_parameters_from_shell(long argc, char **argv);
long  read_train_data();
long  scale_train_data();
long  write_classifier(MCClassifier *mc_cls);
void  print_help();
void  dump(FILE *);

int main(int argc, char **argv) {
  MCClassifier mc_cls;
  long j;
  
  printf("\nMultiClass SVM Train\tversion 1.0\n"); fflush(stdout);
  
  strcpy(mc_cls.comment, *argv);
  for (j=1; j<argc; j++)
    strcat(mc_cls.comment, *(argv+j));

  initialize();
  read_parameters_from_shell(argc, argv);
  read_train_data();
  printf("%ld examples (dimension %ld),  %ld classes\n", mc_datadef.m, mc_datadef.l, mc_datadef.k); fflush(stdout); 
  scale_train_data();
  kernel_text(&spoc_pd.kernel_def, stdout);
#ifdef MONITOR1
  spoc_pd.file_next_p  = ut_fopen("TEMP_NEXT_P","w");
#endif
  mc_cls.solution= spoc(mc_datadef, spoc_pd);
  mc_cls.scale_def = mc_scaledef;
  mc_cls.kernel_def = spoc_pd.kernel_def;
#ifdef MONITOR1
  fclose(spoc_pd.file_next_p);
#endif

  write_classifier(&mc_cls);
  mc_classifier_destruct(&mc_cls);
  spoc_destruct();
  mc_datadef_destruct(&mc_datadef);
  return (0);
}


void initialize() {
  mc_datadef.m = 1;
  mc_datadef.k = 1;
  mc_datadef.l = 1;
  mc_datadef.x = NULL;
  mc_datadef.y = NULL;
  
  spoc_pd.beta = 1e-4;
  spoc_pd.cache_size = 4096;
  
  spoc_pd.kernel_def.kernel_type = KERNEL_EXPONENT_NP;
  spoc_pd.kernel_def.polynom_degree = 1;
  spoc_pd.kernel_def.polynom_a0 = 1;
  spoc_pd.kernel_def.exponent_sigma = .5;
  
  spoc_pd.epsilon =  1e-3;
  spoc_pd.epsilon0 = (1-1e-6);
  spoc_pd.delta =    1e-4;
  spoc_pd.redopt_type = REDOPT_TYPE_EXACT;
  mc_scaledef_initialize(&mc_scaledef);
  
}

long read_parameters_from_shell(long argc, char **argv) {
  char c;

  for (argc--, argv++; argc >0 ; argc-=2, argv++) {
    
    if ((*argv)[0] !='-')
      break;
    c=(*argv)[1];
    argv++;
       
    switch (c) {
    case 'f':
      read_parameters_from_file(*argv);                                               break;
    case 'm':
      mc_datadef.m = atoi(*argv);                                                    break;
    case 'k':
      mc_datadef.k = atoi(*argv);                                                    break;
    case 'l':
      mc_datadef.l = atoi(*argv);                                                    break;
    case 'b':
      spoc_pd.beta = atof(*argv);                                                   break;
    case 'c':
      spoc_pd.cache_size = atol(*argv);                                             break;   
    case 't':
      spoc_pd.kernel_def.kernel_type = (enum KernelType) atoi(*argv);               break;
    case 'd':
      spoc_pd.kernel_def.polynom_degree = atoi(*argv);                              break;
    case 'a':
      spoc_pd.kernel_def.polynom_a0 = atoi(*argv);                                  break;
    case 's':
      spoc_pd.kernel_def.exponent_sigma = atof(*argv);                              break;
    case 'e':  
      spoc_pd.epsilon = atof(*argv);                                                  break;
    case 'z':
      spoc_pd.epsilon0 = atof(*argv);                                                 break;
    case 'w':
      spoc_pd.delta = atof(*argv);                                                    break;
    case 'r':
      spoc_pd.redopt_type = (enum RedOptType)  atoi(*argv);                           break;
    case 'p':
      strcpy(scale_file, *argv); 
      mc_scaledef.to_scale = 1;                                                                   break;
    case 'h':
      print_help(); 
      exit(1);                                                                        break;
    default:
      fprintf(stderr,"bad usage, run mcsvm-train -h for help\n");
      exit(1);
    }
  }
  if (argc != 2) {
    print_help();
    exit(1);
  }

  strcpy(train_data_name, *argv);
  argv++;
  strcpy(classifier_name, *argv);
  return (1);
}

long read_parameters_from_file(char *file_name) {
  
  char line[STRLEN];
  char **table;
  unsigned long i;
  FILE *file;
  

  file= ut_fopen(file_name, "r");
  
  while (fgets(line, STRLEN, file) != NULL) {
    if ((line[0] == '%') || isspace(line[0]))  continue; 
    for (i=0; i<strlen(line); i++)
      if (isspace(line[i])) break;
    if (i<strlen(line))
      line[i]='\0';

    for (table = variables; !((strcmp(*table, "ENDOFTABLE")==0) || (strcmp(*table, line)==0)); table++);
    if  (strcmp(*table, line)==0) {
      if (fgets(line, STRLEN, file) == NULL) {
	fprintf(stderr,"mcsvm-train: missing parameter in file '%s'\n",file_name);
	exit(1);
      }
      if (strcmp(*table, "m")==0)
	mc_datadef.m = atoi(line);
      else if (strcmp(*table, "k")==0)
	mc_datadef.k = atoi(line);
      else if (strcmp(*table, "l")==0)      
	mc_datadef.l = atoi(line);
      else if (strcmp(*table, "beta")==0)
	spoc_pd.beta = atof(line);
      else if (strcmp(*table, "cache_size")==0)
	spoc_pd.cache_size = atol(line);
      else if (strcmp(*table, "kernel_type")==0)
	spoc_pd.kernel_def.kernel_type = (enum KernelType) atoi(line); 
      else if (strcmp(*table, "polynom_degree")==0)
	spoc_pd.kernel_def.polynom_degree = atoi(line);
      else if  (strcmp(*table, "polynom_a0")==0)
	spoc_pd.kernel_def.polynom_a0 = atoi(line);
      else if  (strcmp(*table, "exponent_sigma")==0)
	spoc_pd.kernel_def.exponent_sigma = atof(line);
      else if  (strcmp(*table, "epsilon")==0)	
	spoc_pd.epsilon = atof(line);
      else if  (strcmp(*table, "epsilon0")==0)	
	spoc_pd.epsilon0 = atof(line);
      else if  (strcmp(*table, "delta")==0)
	spoc_pd.delta = atof(line);
      else if  (strcmp(*table, "redopt_type")==0)
	spoc_pd.redopt_type = (enum RedOptType)  atoi(line);
      else if (strcmp(*table, "scale_file")==0) {
	strcpy(scale_file, line);
	mc_scaledef.to_scale = 1;
      }
      else {
	fprintf(stderr,"bad usage, run mcsvm-train -h for help\n");
	exit(1);
      }
    }
    else {
      fprintf(stderr,"bad usage, run mcsvm-train -h for help\n");
      dump(stderr);
      exit(1);
    }
  }
  return (1);
}


  
long read_train_data() {
  FILE *file;
  
  printf("Reading training file '%s' ... " , train_data_name); fflush(stdout);

  
  file = ut_fopen(train_data_name, "r");
  mc_datadef_read(&mc_datadef, file);
  fclose(file);
  printf(" done\n"); fflush(stdout);
  return (1);
}




long write_classifier(MCClassifier *mc_cls) {
  FILE *file;

  printf("Writing classifier file '%s' ... " , classifier_name); fflush(stdout);
  file = ut_fopen(classifier_name,"wb");
  mc_classifier_write(mc_cls, file);
  fclose(file);
  printf("done\n"); fflush(stdout);

#ifdef DEBUG_MCSVM_TRAIN

{
  long r,i;
  for (i=0; i<result_def.n_supp_pattern; i++) {
    for (r=0; r<mc_datadef.k; r++) {
      fprintf(stderr,"%20.10f", result_def.tau[result_def.supp_pattern_list[i]][r]);
    }
    fprintf(stderr,"\n");
  }
}
#endif


  return (1);
}




long  scale_train_data() {
  FILE *file;
  
  if (mc_scaledef.to_scale ==0) {
/*     printf("No scaling\n"); fflush(stdout); */
    return (1);
  }

  printf("Scaling file '%s'\n" , scale_file); fflush(stdout);
  file = ut_fopen(scale_file, "r");
  mc_scaledef_read(&mc_scaledef, file);
  fclose(file);
  mc_scaledef_scale(&mc_scaledef, mc_datadef.x, mc_datadef.m);
  return (1);
}

void dump(FILE *file) {
  fprintf(file, "problem def\n");
  fprintf(file, "\tm :%10ld\n",mc_datadef.m);
  fprintf(file, "\tk :%10ld\n",mc_datadef.k);
  fprintf(file, "\tl :%10ld\n",mc_datadef.l);
  fprintf(file, "\tx :%p\n",mc_datadef.x);
  fprintf(file, "\ty :%p\n",mc_datadef.y);
  
  fprintf(file, "spoc_pd\n");
  fprintf(file, "\tbeta :%10.5e\n",spoc_pd.beta);
  fprintf(file, "\tcache_size :%ld\n",spoc_pd.cache_size);
  
  fprintf(file, "kernel\n");
  fprintf(file, "\tkernel_type :\t%s\n",kernel_get_type_name(spoc_pd.kernel_def.kernel_type));
  fprintf(file, "\tpolynom_degree :\t%ld\n",spoc_pd.kernel_def.polynom_degree);
  fprintf(file, "\tpolynom_a0 :\t%f\n",spoc_pd.kernel_def.polynom_a0);
  fprintf(file, "\texponent_sigma :\t%f\n",spoc_pd.kernel_def.exponent_sigma);
  
  fprintf(file, "run def\n");
  fprintf(file, "\tepsilon :%10.5e\n",spoc_pd.epsilon);
  fprintf(file, "\tepsilon0 :%10.5e\n",spoc_pd.epsilon0);
  fprintf(file, "\tdelta :%10.5e\n",spoc_pd.delta); 
  fprintf(file, "\tred_opt_type :%d\n",spoc_pd.redopt_type);
  
}

void print_help() {
  fprintf(stderr, "Written by Koby Crammer, Hebrew Univerity of Jerusalem, Jan 2003\n");
  fprintf(stderr, "kobics@cs.huji.ac.il\n");
  fprintf(stderr, "\n");
  fprintf(stderr, "Usage : mcsvm-train [options] [train data] [classifier name]\n");
  fprintf(stderr, "\n");
  fprintf(stderr, "train data :\t\ttraining data file.\n");
  fprintf(stderr, "\t\t\teach example(line) contains label and an instance vector\n");
  fprintf(stderr, "classifier name :\tclassifier parameters file\n");  
  fprintf(stderr, "\n");
  fprintf(stderr, "Options\n");
  fprintf(stderr, "-m (int)\t\tno. of training examples [m]\n");
  fprintf(stderr, "-l (int) \t\tdata dimension [l]\n");
  fprintf(stderr, "-k (int) \t\tno. of classes [k]\n");
  fprintf(stderr, "-b (double) \t\tmargin [beta]\n");
  fprintf(stderr, "-e (double)\t\ttolerance value [epsilon]\n");
  fprintf(stderr, "-z (double)\t\tinitialize margin [epsilon0]\n");
  fprintf(stderr, "-c (int)\t\tmaximal [cache_size](Mb)\n");
  fprintf(stderr, "-t (0..4)\t\tkernel type:[kernel_type]\n");
  fprintf(stderr, "\t\t\t0 - exponent exp(-(||A-B||^2)/(2*sigma^2))\n");
  fprintf(stderr, "\t\t\t1 - exponent np (sigma=1)\n");
  fprintf(stderr, "\t\t\t2 - homogeneous polynom (AxB)^degree\n");
  fprintf(stderr, "\t\t\t3 - non-homogeneous polynom (a0+AxB)^degree\n");
  fprintf(stderr, "\t\t\t4 - non-homogeneous polynom (1+AxB)^degree\n");
  fprintf(stderr, "-d (int)\t\tpolynomial degree [polynom_degree]\n"); 
  fprintf(stderr, "-a (int)\t\tpolynomial constant a0 [polynom_a0]\n"); 
  fprintf(stderr, "-s (int)\t\texponent standard deviation [exponent_sigma]\n");
  fprintf(stderr, "-r (int)\t\treduced optimizer [redopt_type]:\n");
  fprintf(stderr, "\t\t\t0 - exact\n");
  fprintf(stderr, "\t\t\t1 - approximate\n");
  fprintf(stderr, "\t\t\t2 - binary exact\n"); 
  fprintf(stderr, "-w (double)\t\tapproximation tolerance [delta] for approximate method\n");
  fprintf(stderr, "-p (string)\t\tdefinition file. use [keywords] above, then value in new line\n");
  fprintf(stderr, "-h\t\t\tthis help\n\n");
  fprintf(stderr, "References:\n");
  fprintf(stderr, "[1]\tOn the Algorithmic Implementation of Multiclass Kernel-based Vector Machines,\n\tKoby Crammer and Yoram Singer,\n\tJournal of Machine Learning Research, 2001.\n");
  fprintf(stderr, "[2]\tUltraconservative Online Algorithms for Multiclass Problems,\n\tKoby Crammer and Yoram Singer,\n\tJournal of Machine Learning Research, 2003.\n"); fprintf(stderr, "[3]\tOn the Learnability and Design of Output Codes for Multiclass Problems,\n\tKoby Crammer and Yoram Singer,\n\tMachine Learning 47, 2002.");


}










/*  =========================== */
