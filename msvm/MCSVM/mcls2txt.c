/* $Author: kobics $ */
/* $Date: 2003/01/30 12:44:05 $ */
/* $Source: /cs/phd/kobics/.CVSROOT/code/multiClass/mcls2txt.c,v $ */
/* $Name:  $ */
/* $Locker:  $ */
/* $Revision: 5.5 $ */
/* $State: Exp $ */



#include "mucutils.h"
#include "utilities.h"
  
#define STRLEN 256
int main(int argc, char **argv) {
  FILE *file;
  MCClassifier mc_cls;
  
  if (argc != 2) {
    printf("\nMultiClass to Text\tversion 1.0\n"); fflush(stdout);
    fprintf(stderr, "Copyright: Koby Crammer, Hebrew Univerity of Jerusalem, Jan 2003\n");
    fprintf(stderr, "kobics@cs.huji.ac.il\n");
    fprintf(stderr, "\nusage mcls2txt [classifier name]\n");
    exit(1);
  }
  
  file = ut_fopen(*++argv, "rb");
  mc_classifier_read(&mc_cls, file);
  ut_fclose(file);
  mc_classifier_text(&mc_cls, stdout);
  mc_solution_destruct(&mc_cls.solution);
  return (0);
}
