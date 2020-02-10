# $Author: kobics $ */
# $Date: 2003/08/06 11:19:40 $ */
# $Source: /cs/phd/kobics/.CVSROOT/code/multiClass/Makefile,v $ */
# $Name:  $ */
# $Locker:  $ */
# $Revision: 6.10 $ */
# $State: Exp $ */

  




LDFLAGS=-Wall -W -g
LDLIBS=-lm
CFLAGS=-Wall -W -g
CC = gcc -g
targets= mcsvm-train mcls2txt mcol-test mcol-train mcsvm-ol-train mcall-train


all: $(targets)
	@echo all done


mcsvm-train: cachelru.o kernel.o spoc.o redopt.o utilities.o mucutils.o mcdata.o 
mcls2txt: utilities.o mucutils.o kernel.o
mcol-train: cachelru.o kernel.o redopt.o utilities.o mucutils.o mira.o mcdata.o 
mcall-train: cachelru.o kernel.o redopt.o utilities.o mucutils.o mconline.o mcdata.o 
mcol-test: kernel.o utilities.o mucutils.o mcdata.o 
mcsvm-ol-train: cachelru.o kernel.o spoc.o redopt.o utilities.o mucutils.o mira.o mcdata.o

clean:
	rm -f *.o gmon.out mcsvm-train mcsvm-ol-train mcol-train mcol-test mcls2txt mcall-train
tild:
	mv -f *~ tild

cachelru:
kernel:
scale:
spoc:
redopt:
mcsvm-utils:
mucutils: 
mira:
mcall:
mcol-train:
mcall-train:
mcdata:

