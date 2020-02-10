/* $Author: kobics $ */
/* $Date: 2003/06/26 07:21:18 $ */
/* $Source: /cs/phd/kobics/.CVSROOT/code/multiClass/mcall-train.c,v $ */
/* $Name:  $ */
/* $Locker:  $ */
/* $Revision: 6.4 $ */
/* $State: Exp $ */



#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include "utilities.h"
#include "mucutils.h"
#include "mconline.h"

#define STRLEN 256

static MCScaleDef mc_scaledef;
static MCDataDef mc_datadef;
static MCOnlineParamDef mconline_pd;

static char *variables[] = {
  "m", "k", "l",
  "beta",
  "kernel_type", "polynom_degree", "polynom_a0", "exponent_sigma",
  "epochs", "is_voted", "delta", "redopt_type",
  "scale_file",
  "ENDOFTABLE"
};

static char train_data_name[STRLEN];
static char classifier_name[STRLEN];
static char init_classifier_name[STRLEN] = "";
static char scale_file[STRLEN];


void initialize();
long read_parameters_from_file(char *file_name);
long read_parameters_from_shell(long argc, char **argv);
long read_train_data();
long scale_train_data();
void read_classifier(char *file_name, MCClassifier *mc_cls_p);
long write_classifier(MCClassifier *mc_cls);
void print_help();
void modify_values(MCClassifier *mc_cls_p);
void dump(FILE *);


int main(int argc, char **argv) {
  MCClassifier mc_cls;
  MCClassifier* mc_cls_init= NULL;
  long j;

  printf("\nMultiClass OnLine Train (budget)\tversion 1.3\n"); fflush(stdout);

  strcpy(mc_cls.comment, *argv);
  for (j=1; j<argc; j++) {
    strcat(mc_cls.comment, *(argv+j));
    strcat(mc_cls.comment, " ");
  }

  initialize();
  read_parameters_from_shell(argc, argv);
  if (strlen(init_classifier_name) > 0) {
    mc_cls_init = ut_calloc(1, sizeof(MCClassifier));
    mc_scaledef_initialize(&mc_cls_init->scale_def);
    read_classifier(init_classifier_name, mc_cls_init);
    modify_values(mc_cls_init);
  }

  read_train_data();
  printf("%ld examples (dimension %ld),  %ld classes\n", mc_datadef.m, mc_datadef.l, mc_datadef.k); fflush(stdout); 
  scale_train_data();
  kernel_text(&mconline_pd.kernel_def, stdout);


  mc_cls.solution = mconline(mc_datadef, mconline_pd, mc_cls_init);

  mc_cls.solution.size = mc_datadef.m;
  mc_cls.solution.k    = mc_datadef.k;
  mc_cls.solution.l    = mc_datadef.l;

  mc_cls.scale_def = mc_scaledef;
  mc_cls.kernel_def = mconline_pd.kernel_def;
  
  write_classifier(&mc_cls);
  
  mc_datadef_destruct(&mc_datadef);
  mc_classifier_destruct(&mc_cls);

  if (strlen(init_classifier_name) > 0) {
    mc_classifier_destruct(mc_cls_init);
    free (mc_cls_init);
  }

  return (1);
}


void initialize() {
  mc_datadef.m = 1;
  mc_datadef.k = 1;
  mc_datadef.l = 1;
  mc_datadef.x = NULL;
  mc_datadef.y = NULL;
  
  mconline_pd.beta =1;
  mconline_pd.epochs =1;
  mconline_pd.is_voted =0;

  mconline_pd.kernel_def.kernel_type = KERNEL_EXPONENT_NP;
  mconline_pd.kernel_def.polynom_degree = 1;
  mconline_pd.kernel_def.polynom_a0 = 1;
  mconline_pd.kernel_def.exponent_sigma = .5;

  mconline_pd.redopt_type =  REDOPT_TYPE_EXACT;
  mconline_pd.delta = 1e-4;
  
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
      mc_datadef.m = atoi(*argv);                                                     break;
    case 'k':
      mc_datadef.k = atoi(*argv);                                                     break;
    case 'l':
      mc_datadef.l = atoi(*argv);                                                     break;
    case 'b':
      mconline_pd.beta = atof(*argv);                                                 break;
    case 'c':
      mconline_pd.spp_pattern_size = atoi(*argv);                                      break;
    case 'u':
      mconline_pd.update_type = (enum MCOnlineUpdateType) atoi(*argv);                break;
    case 'j':
      mconline_pd.stage_type = (enum MCOnlineStageType) atoi(*argv);                  break;
    case 'i':
      mconline_pd.find_type = (enum MCOnlineFindType) atoi(*argv);                    break;
    case 'x':
      mconline_pd.gamma = atof(*argv);                                                    break;
    case 't':
      mconline_pd.kernel_def.kernel_type = (enum KernelType) atoi(*argv);                 break;
    case 'd':
      mconline_pd.kernel_def.polynom_degree = atoi(*argv);                                break;
    case 'a':
      mconline_pd.kernel_def.polynom_a0 = atof(*argv);                                    break;
    case 's':
      mconline_pd.kernel_def.exponent_sigma = atof(*argv);                                break;
    case 'w':
      mconline_pd.delta = atof(*argv);                                                    break;
    case 'r':
      mconline_pd.redopt_type = (enum RedOptType)  atoi(*argv);                           break;
    case 'g':
      mconline_pd.epochs = atof(*argv);                                                   break;  
    case 'q':
      mconline_pd.is_voted = atoi(*argv);                                                break;
    case 'p':
      strcpy(scale_file, *argv); 
      mc_scaledef.to_scale = 1;                                                       break;
    case 'n':
      strcpy(init_classifier_name, *argv);                                            break;
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
	fprintf(stderr,"mcall-train: missing parameter in file '%s'\n",file_name);
	exit(1);
      }
      if (strcmp(*table, "m")==0)
	mc_datadef.m = atoi(line);
      else if (strcmp(*table, "k")==0)
	mc_datadef.k = atoi(line);
      else if (strcmp(*table, "l")==0)      
	mc_datadef.l = atoi(line);
      else if (strcmp(*table, "beta")==0)
	mconline_pd.beta = atof(line);
      else if (strcmp(*table, "spp_pattern_size")==0)
	mconline_pd.spp_pattern_size = atoi(line);                
      else if (strcmp(*table, "update_type")==0)
	mconline_pd.update_type = (enum MCOnlineUpdateType) atoi(line);     
      else if (strcmp(*table, "stage_type")==0)
	mconline_pd.stage_type = (enum MCOnlineStageType) atoi(line);       
      else if (strcmp(*table, "find_type")==0)
	mconline_pd.find_type = (enum MCOnlineFindType) atoi(line);       
      else if (strcmp(*table, "gamma")==0)
	mconline_pd.gamma = atof(line);       
      else if (strcmp(*table, "kernel_type")==0)
	mconline_pd.kernel_def.kernel_type = (enum KernelType) atoi(line); 
      else if (strcmp(*table, "polynom_degree")==0)
	mconline_pd.kernel_def.polynom_degree = atoi(line);
      else if  (strcmp(*table, "polynom_a0")==0)
	mconline_pd.kernel_def.polynom_a0 = atoi(line);
      else if  (strcmp(*table, "exponent_sigma")==0)
	mconline_pd.kernel_def.exponent_sigma = atof(line);
      else if (strcmp(*table, "epochs")==0)
	mconline_pd.epochs = atof(line);   
      else if (strcmp(*table, "is_voted")==0)
	mconline_pd.is_voted = atoi(line);
      else if  (strcmp(*table, "delta")==0)
	mconline_pd.delta = atof(line);
      else if  (strcmp(*table, "redopt_type")==0)
	mconline_pd.redopt_type = (enum RedOptType)  atoi(line);
      else if (strcmp(*table, "scale_file")==0) {
	strcpy(scale_file, line);
	mc_scaledef.to_scale = 1;
      }
      else if (strcmp(*table, "init_classifer_name")==0) 
	strcpy(init_classifier_name, line);
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

  printf("Readind training file '%s' ... " , train_data_name); fflush(stdout);
  
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
  /*   printf("No scaling\n"); fflush(stdout); */
    return (1);
  }

  printf("Scaling file '%s'\n" , scale_file); fflush(stdout);
  file = ut_fopen(scale_file, "rb");
  mc_scaledef_read(&mc_scaledef, file);
  fclose(file);
  mc_scaledef_scale(&mc_scaledef, mc_datadef.x, mc_datadef.m);
  return (1);
}


void read_classifier(char *file_name, MCClassifier *mc_cls_p) {
  FILE *file;
  
  printf("Reading classifier file '%s' ... ", file_name); fflush(stdout);
  
  file = ut_fopen(file_name, "rb");
  mc_classifier_read(mc_cls_p, file);
  fclose(file);
  
  printf("done\n");


}

void modify_values(MCClassifier *mc_cls_p) {
  long i;

  mc_datadef.m = mc_cls_p->solution.size;
  mc_datadef.k = mc_cls_p->solution.k;
  mc_datadef.l = mc_cls_p->solution.l;
  mconline_pd.is_voted = mc_cls_p->solution.is_voted;
  
  mconline_pd.kernel_def = mc_cls_p->kernel_def;
  
  if (mc_cls_p->solution.n_supp_pattern > mconline_pd.spp_pattern_size) {
    fprintf(stderr, "No. already support patterns (%ld) > bound support pattern (%ld)\n", mc_cls_p->solution.n_supp_pattern, mconline_pd.spp_pattern_size);
    exit(0);
  }
  
  mc_scaledef.to_scale     = mc_cls_p->scale_def.to_scale;
  mc_scaledef.l            = mc_cls_p->scale_def.l;
  mc_scaledef.scale_factor = mc_cls_p->scale_def.scale_factor;
  mc_scaledef.to_zero_data_mean = mc_cls_p->scale_def.to_zero_data_mean;
  if (mc_scaledef.data_mean != NULL) {
    free (mc_scaledef.data_mean);
    mc_scaledef.data_mean = NULL;
  }
  if (mc_scaledef.to_zero_data_mean == 1) {
    mc_scaledef.data_mean = (double *) ut_calloc(mc_scaledef.l, sizeof(double));
    for (i=0; i<mc_scaledef.l; i++)
      mc_scaledef.data_mean[i] = mc_cls_p->scale_def.data_mean[i];
  }
}  


void dump(FILE *file) {
  fprintf(file, "MCDataDef\n");
  fprintf(file, "\tm :%10ld\n",mc_datadef.m);
  fprintf(file, "\tk :%10ld\n",mc_datadef.k);
  fprintf(file, "\tl :%10ld\n",mc_datadef.l);
  fprintf(file, "\tx :%p\n",mc_datadef.x);
  fprintf(file, "\ty :%p\n",mc_datadef.y);
  
  fprintf(file, "MCOnlineParamDef\n");
  fprintf(file, "\tbeta :%10.5e\n",mconline_pd.beta);
  fprintf(file, "\teopchs :%f\n",mconline_pd.epochs);
  fprintf(file, "\tis_voted :%ld\n",mconline_pd.is_voted);
  fprintf(file, "\tdelta :%10.5e\n",mconline_pd.delta); 
  fprintf(file, "\tred_opt_type :%d\n",mconline_pd.redopt_type);
  
  fprintf(file, "Kernel\n");
  fprintf(file, "\tkernel_type :\t%s\n",kernel_get_type_name(mconline_pd.kernel_def.kernel_type));
  fprintf(file, "\tpolynom_degree :\t%ld\n",mconline_pd.kernel_def.polynom_degree);
  fprintf(file, "\tpolynom_a0 :\t%f\n",mconline_pd.kernel_def.polynom_a0);
  fprintf(file, "\texponent_sigma :\t%f\n",mconline_pd.kernel_def.exponent_sigma);
}


void print_help() {
  fprintf(stderr, "Copyright: Koby Crammer, Jan 2001\n");
  fprintf(stderr, "\n");
  fprintf(stderr, "Usage : mcol-train [options] [train data] [classifier name]\n");
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
  fprintf(stderr, "-g (double) \t\t[epochs]\n");
  fprintf(stderr, "-c (int) \t\tspp pattern size[example_set_size]\n");
  fprintf(stderr, "-u (0..7) \t\tupdate type:[update_type]\n");
  fprintf(stderr, "\t\t\t0 - uniform\n");
  fprintf(stderr, "\t\t\t1 - max\n");
  fprintf(stderr, "\t\t\t2 - prop\n");
  fprintf(stderr, "\t\t\t3 - mira\n");
  fprintf(stderr, "\t\t\t4 - perceptron\n");
  fprintf(stderr, "\t\t\t5 - 1-vs-rest mira\n");
  fprintf(stderr, "\t\t\t6 - mira (not in use)\n");
  fprintf(stderr, "\t\t\t7 - rand\n");
  fprintf(stderr, "-j (0..3) \t\tstage type:[replace_type]\n");
  fprintf(stderr, "\t\t\t0 - bound-find-update\n");
  fprintf(stderr, "\t\t\t1 - update-bound-find\n");
  fprintf(stderr, "\t\t\t2 - update-find-all\n");
  fprintf(stderr, "\t\t\t3 - update-find-one\n");
  fprintf(stderr, "-i (0..4) \t\tfind type:[find_type]\n");
  fprintf(stderr, "\t\t\t0 - maximal margin \n");
  fprintf(stderr, "\t\t\t1 - minimal weight \n");
  fprintf(stderr, "\t\t\t2 - maximal margin (w/o example) (CC-I)\n");
  fprintf(stderr, "\t\t\t3 - minimal norm (CC-II)\n");
  fprintf(stderr, "\t\t\t4 - minimal norm (normalized) (CC-II)\n");
  fprintf(stderr, "-x (double) \t\tremove threshold [gamma]\n");
/*   fprintf(stderr, "-q (0,1) \t\t[is_voted]\n"); */
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
  fprintf(stderr, "-n (string)\t\tinit classifier file\n");
  fprintf(stderr, "-w (double)\t\tapproximation tolerance [delta] for approximate method\n");
  fprintf(stderr, "-f (string)\t\tdefinition file. use [keywords] above, then value in new line\n");
/*   fprintf(stderr, "-n (string)\t\tscale file.\n"); */
  fprintf(stderr, "-h\t\t\tthis help\n");
}
