/* $Author: kobics $ */
/* $Date: 2003/10/01 11:49:09 $ */
/* $Source: /cs/phd/kobics/.CVSROOT/code/multiClass/mcol-test.c,v $ */
/* $Name:  $ */
/* $Locker:  $ */
/* $Revision: 6.10.6.1 $ */
/* $State: Exp $ */



#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <time.h>
#include <string.h>
#include "utilities.h"
#include "mucutils.h"
  
#define STRLEN 256
#define ROUND_PER 10

typedef struct {
  long n_errors;
  long **error_statistics;
} ErrStatistics;

static MCDataDef   datadef_train;
static MCDataDef   datadef_test;
static MCClassifier mc_cls; 

static long **supp_tau_lists;
static long *n_supp_tau;

static char train_data_name[STRLEN];
static char classifier_name[STRLEN];
static char test_data_name[STRLEN];
static char report_name[STRLEN];

static ErrStatistics error_test;
static KF kernel_fun;

void read_classifier(char *file_name);
void read_data(char *file_name, MCDataDef *pd);
void read_test_data(char *file_name);
void initialize();
void data_errors(ErrStatistics *es);

void error_report(char *file_name);
void free_memory();


int main(int argc, char **argv) {
  clock_t data_error_time0, data_error_time1; 
  
  printf("\nMultiClass Test  version 1.0\n"); fflush(stdout);

  
  if (argc != 6) {
    fprintf(stderr, "Written by Koby Crammer, Hebrew Univerity of Jerusalem, Jan 2003\n");
    fprintf(stderr, "kobics@cs.huji.ac.il\n");
    fprintf(stderr, "usage mcol-test [train data] [classifier] [test data] [report] [no. test data]\n");
    exit(1);
  }
  strcpy(train_data_name, *++argv);
  strcpy(classifier_name, *++argv);
  strcpy(test_data_name, *++argv);
  strcpy(report_name, *++argv);
  
  mc_scaledef_initialize(&mc_cls.scale_def);
  read_classifier(classifier_name);
  kernel_construct(mc_cls.solution.l);

  datadef_train.m = mc_cls.solution.size;
  datadef_train.k = mc_cls.solution.k;
  datadef_train.l = mc_cls.solution.l;
  
  datadef_test.m = atoi(*++argv);
  datadef_test.k = mc_cls.solution.k;
  datadef_test.l = mc_cls.solution.l;
  
  printf("Training file '%s' (%ld points)\n", train_data_name,  datadef_train.m);
  printf("Testing  file '%s' (%ld points)\n", test_data_name,   datadef_test.m);
  printf("Data dimension %ld\n", mc_cls.solution.l);
  printf("No. of classes %ld\n",  mc_cls.solution.k);
  kernel_text(&mc_cls.kernel_def, stdout);
  if (mc_cls.scale_def.to_scale ==0) {
    /*     printf("No scaling\n"); fflush(stdout); */
  }
  
  read_data(train_data_name, &datadef_train);
  read_data(test_data_name,  &datadef_test);
  initialize();
  kernel_fun = kernel_get_function(mc_cls.kernel_def);

  data_error_time0 = clock();
  data_errors(&error_test);
  data_error_time1 = clock();
  error_report(report_name);
  free_memory();
/*   printf("\nMCSVM Test Error time: %20.6f(sec)\n", ((double)(data_error_time1 - data_error_time0))/CLOCKS_PER_SEC); */

  return (1);
}


void read_classifier(char *file_name) {
  FILE *file;
  
  printf("Reading classifier file '%s' ... ", classifier_name); fflush(stdout);

  file = ut_fopen(file_name, "rb");
  mc_classifier_read(&mc_cls, file);
  fclose(file);

  printf("done\n");


}


void read_data(char *file_name, MCDataDef *pd) {
  FILE *file;

  
  printf("Reading data file '%s' ... ", file_name); fflush(stdout);
  
  file = ut_fopen(file_name, "r");
  mc_datadef_read(pd, file);
  fclose(file);
  mc_scaledef_scale(&mc_cls.scale_def, pd->x, pd->m);
  
  printf("done\n");fflush(stdout);
}

void initialize() {
  long r;
  long si;
   
  n_supp_tau = (long *) ut_calloc(mc_cls.solution.k, sizeof(long));
  
  for (r=0; r<mc_cls.solution.k; r++)
    n_supp_tau[r] = 0;
  
  supp_tau_lists = (long **) ut_calloc(mc_cls.solution.k, sizeof(long*));
  *supp_tau_lists = (long *) ut_calloc(mc_cls.solution.n_supp_pattern * mc_cls.solution.k, sizeof(long));
  for (r=1; r<mc_cls.solution.k; r++) 
    supp_tau_lists[r] = supp_tau_lists[r-1] + mc_cls.solution.n_supp_pattern;
  
  for (r=0; r<mc_cls.solution.k; r++) {
    for (si=0; si<mc_cls.solution.n_supp_pattern; si++) {
      if (mc_cls.solution.tau[si][r] != 0) {
	supp_tau_lists[r][n_supp_tau[r]++] = si;
      }
    }
  }

  
  printf("Total support patterns %ld\n\n", mc_cls.solution.n_supp_pattern);
  printf("\t\tclass\tsupport patterns per class\n");
  printf("\t\t-----\t--------------------------\n");
  for (r=0; r<mc_cls.solution.k; r++)
    printf("\t\t  %ld\t    %ld\n", r, n_supp_tau[r]);
  fflush(stdout);
}




void data_errors(ErrStatistics *es) {

  long round;
  long i, si;
  long r, s;
  double *kernel_values = NULL;
  double sim_score;
  double max_sim_score;
  long n_max_sim_score;
  long best_y;
  long supp_pattern_index;

  printf("\nCumulative test error (every 10%% of data points)\n"); fflush(stdout);


  es->n_errors =0;
  kernel_values = (double *) ut_calloc(mc_cls.solution.n_supp_pattern, sizeof(double));
  
  es->error_statistics = (long **) ut_calloc(mc_cls.solution.k, sizeof(long*));
  *es->error_statistics = (long *) ut_calloc(mc_cls.solution.k * mc_cls.solution.k, sizeof(long));
  
  for (r=1; r<mc_cls.solution.k; r++) 
    es->error_statistics[r] = es->error_statistics[r-1] + datadef_test.k;

  for (r=0; r<mc_cls.solution.k; r++)
    for (s=0; s<mc_cls.solution.k; s++)
      es->error_statistics[r][s] =0;


  round = datadef_test.m / ROUND_PER;
  for (i=0; i<datadef_test.m; i++) {
    for (si=0; si<mc_cls.solution.n_supp_pattern; si++)
      kernel_values[si] = kernel_fun(datadef_test.x[i], datadef_train.x[mc_cls.solution.supp_pattern_list[si]]);
    
    n_max_sim_score =0;
    max_sim_score = -DBL_MAX;
    best_y = -1;
    
    for (r=0; r<datadef_test.k; r++) {
      sim_score =0;
      for (si=0; si<n_supp_tau[r]; si++) {
	supp_pattern_index = supp_tau_lists[r][si];
	sim_score += mc_cls.solution.tau[supp_pattern_index][r] * kernel_values[supp_pattern_index];
      }
      
      if (sim_score > max_sim_score) {
	max_sim_score = sim_score;
	n_max_sim_score =1;
	best_y = r;
      }
      else if (sim_score == max_sim_score)
	n_max_sim_score++;
    }
    
    es->error_statistics[datadef_test.y[i]][best_y]++;
    if ((n_max_sim_score>1) || (best_y != datadef_test.y[i])) {
      es->n_errors++;
    }

    if ((i % round) == (round - 1)) {
      printf("%8ld - %4.2f%% (%ld / %ld)\n", i+1, 
	      100*((double)error_test.n_errors / (double)(i+1)) ,
	      error_test.n_errors ,
	      i+1);
    fflush(stdout);
    }
  }
  free(kernel_values);
    
}


void error_report(char *file_name) {
  FILE *file;
  long r, s;
  long t;

  
  printf("Writing report file '%s' ... ", file_name); fflush(stdout);
    
  file = ut_fopen(file_name, "w");
  
  fprintf(file, "\nSummary\n");
  fprintf(file, "=======================\n");
  
  fprintf(file, "\nGeneral\n");
  fprintf(file, " no of labels : %ld\n", datadef_test.k);
  fprintf(file, " dimension    : %ld\n", datadef_test.l);

  fprintf(file, "\nFiles\n");
  fprintf(file, " train data : %s\t(%ld examples)\n", train_data_name, datadef_train.m);
  fprintf(file, " classifier : %s\n", classifier_name);
  fprintf(file, " test data  : %s\t(%ld examples)\n", test_data_name, datadef_test.m);

  fprintf(file, "\nKernel\n");
  fprintf(file, " kernel_type :\t%s\n" ,kernel_get_type_name(mc_cls.kernel_def.kernel_type));
  fprintf(file, " polynom_degree :\t%ld\n" ,mc_cls.kernel_def.polynom_degree);
  fprintf(file, " polynom_a0 :\t%f\n" ,mc_cls.kernel_def.polynom_a0);
  fprintf(file, " exponent_sigma :\t%f\n" ,mc_cls.kernel_def.exponent_sigma);
  
  fprintf(file,"\nScaling\n");
  fprintf(file," used scaling : %s\n", (mc_cls.scale_def.to_scale==0) ? "no" : "yes");
  fprintf(file," scale factor : %.10f\n", mc_cls.scale_def.scale_factor);
  fprintf(file," zero mean    : %s\n", (mc_cls.scale_def.to_zero_data_mean==0) ? "no" : "yes");
  if (mc_cls.scale_def.to_zero_data_mean==1) {
    fprintf(file," mean : \n");
    for (t=0; t<datadef_test.l; t++) {
      if (t%4==3) fprintf(file,"\n");
      fprintf(file, "%10.5f ", mc_cls.scale_def.data_mean[t]);
    }
  }

  fprintf(file, "\nSupport Patterns\n");
  fprintf(file, " total suppurt patterns : %ld\n", mc_cls.solution.n_supp_pattern);
  fprintf(file, " suppurt patterns per class\n");
  fprintf(file, "\t\tclass\tsupport patterns\n");
  for (r=0; r<datadef_train.k; r++)
    fprintf(file, "\t\t%ld\t%ld\n", r, n_supp_tau[r]);
   
  fprintf(file, "\nError Statsitics\n");
  fprintf(file, " test error : %4.2f%% (%ld / %ld)\n\n",
	  100*((double)error_test.n_errors / (double)datadef_test.m) ,
	  error_test.n_errors ,
	  datadef_test.m);
  
  fprintf(file," error statistics (correct/predicted)\n");
  
  fprintf(file,"     ");
  for (r=0; r<datadef_test.k; r++)
    fprintf(file,"%4ld ",r);
  fprintf(file,"\n");
  for (r=0; r<datadef_test.k; r++) {
    fprintf(file,"%4ld ",r);
    for (s=0; s<datadef_test.k; s++)
      fprintf(file,"%4ld ",error_test.error_statistics[r][s]);
    fprintf(file,"\n");
  }
  fprintf(file,"\n");
  fclose(file);
  
  printf("done\n"); fflush(stdout);
}

void free_memory() {
  mc_datadef_destruct(&datadef_train);
  mc_datadef_destruct(&datadef_test);
  mc_classifier_destruct(&mc_cls);
    
  free(n_supp_tau);
  free(*supp_tau_lists);
  free(supp_tau_lists);

  free(*error_test.error_statistics);
  free(error_test.error_statistics);

}
