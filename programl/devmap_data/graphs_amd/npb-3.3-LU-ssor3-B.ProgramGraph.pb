

[external]
KcallBC
A
	full_text4
2
0%7 = tail call i64 @_Z13get_global_idj(i32 2) #3
,addB%
#
	full_text

%8 = add i64 %7, 1
"i64B

	full_text


i64 %7
4truncB+
)
	full_text

%9 = trunc i64 %8 to i32
"i64B

	full_text


i64 %8
LcallBD
B
	full_text5
3
1%10 = tail call i64 @_Z13get_global_idj(i32 1) #3
.addB'
%
	full_text

%11 = add i64 %10, 1
#i64B

	full_text
	
i64 %10
LcallBD
B
	full_text5
3
1%12 = tail call i64 @_Z13get_global_idj(i32 0) #3
.addB'
%
	full_text

%13 = add i64 %12, 1
#i64B

	full_text
	
i64 %12
2addB+
)
	full_text

%14 = add nsw i32 %5, -1
5icmpB-
+
	full_text

%15 = icmp sgt i32 %14, %9
#i32B

	full_text
	
i32 %14
"i32B

	full_text


i32 %9
8brB2
0
	full_text#
!
br i1 %15, label %16, label %58
!i1B

	full_text


i1 %15
8trunc8B-
+
	full_text

%17 = trunc i64 %13 to i32
%i648B

	full_text
	
i64 %13
8trunc8B-
+
	full_text

%18 = trunc i64 %11 to i32
%i648B

	full_text
	
i64 %11
4add8B+
)
	full_text

%19 = add nsw i32 %4, -1
8icmp8B.
,
	full_text

%20 = icmp sgt i32 %19, %18
%i328B

	full_text
	
i32 %19
%i328B

	full_text
	
i32 %18
4add8B+
)
	full_text

%21 = add nsw i32 %3, -1
8icmp8B.
,
	full_text

%22 = icmp sgt i32 %21, %17
%i328B

	full_text
	
i32 %21
%i328B

	full_text
	
i32 %17
1and8B(
&
	full_text

%23 = and i1 %20, %22
#i18B

	full_text


i1 %20
#i18B

	full_text


i1 %22
:br8B2
0
	full_text#
!
br i1 %23, label %24, label %58
#i18B

	full_text


i1 %23
Ybitcast8BL
J
	full_text=
;
9%25 = bitcast double* %0 to [103 x [103 x [5 x double]]]*
Ybitcast8BL
J
	full_text=
;
9%26 = bitcast double* %1 to [103 x [103 x [5 x double]]]*
0shl8B'
%
	full_text

%27 = shl i64 %8, 32
$i648B

	full_text


i64 %8
9ashr8B/
-
	full_text 

%28 = ashr exact i64 %27, 32
%i648B

	full_text
	
i64 %27
1shl8B(
&
	full_text

%29 = shl i64 %11, 32
%i648B

	full_text
	
i64 %11
9ashr8B/
-
	full_text 

%30 = ashr exact i64 %29, 32
%i648B

	full_text
	
i64 %29
1shl8B(
&
	full_text

%31 = shl i64 %13, 32
%i648B

	full_text
	
i64 %13
9ashr8B/
-
	full_text 

%32 = ashr exact i64 %31, 32
%i648B

	full_text
	
i64 %31
�getelementptr8B�
�
	full_text�
�
~%33 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %25, i64 %28, i64 %30, i64 %32, i64 0
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %25
%i648B

	full_text
	
i64 %28
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Nload8BD
B
	full_text5
3
1%34 = load double, double* %33, align 8, !tbaa !8
-double*8B

	full_text

double* %33
�getelementptr8B�
�
	full_text�
�
~%35 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %26, i64 %28, i64 %30, i64 %32, i64 0
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %26
%i648B

	full_text
	
i64 %28
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Nload8BD
B
	full_text5
3
1%36 = load double, double* %35, align 8, !tbaa !8
-double*8B

	full_text

double* %35
hcall8B^
\
	full_textO
M
K%37 = tail call double @llvm.fmuladd.f64(double %2, double %36, double %34)
+double8B

	full_text


double %36
+double8B

	full_text


double %34
Nstore8BC
A
	full_text4
2
0store double %37, double* %33, align 8, !tbaa !8
+double8B

	full_text


double %37
-double*8B

	full_text

double* %33
�getelementptr8B�
�
	full_text�
�
~%38 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %25, i64 %28, i64 %30, i64 %32, i64 1
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %25
%i648B

	full_text
	
i64 %28
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Nload8BD
B
	full_text5
3
1%39 = load double, double* %38, align 8, !tbaa !8
-double*8B

	full_text

double* %38
�getelementptr8B�
�
	full_text�
�
~%40 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %26, i64 %28, i64 %30, i64 %32, i64 1
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %26
%i648B

	full_text
	
i64 %28
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Nload8BD
B
	full_text5
3
1%41 = load double, double* %40, align 8, !tbaa !8
-double*8B

	full_text

double* %40
hcall8B^
\
	full_textO
M
K%42 = tail call double @llvm.fmuladd.f64(double %2, double %41, double %39)
+double8B

	full_text


double %41
+double8B

	full_text


double %39
Nstore8BC
A
	full_text4
2
0store double %42, double* %38, align 8, !tbaa !8
+double8B

	full_text


double %42
-double*8B

	full_text

double* %38
�getelementptr8B�
�
	full_text�
�
~%43 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %25, i64 %28, i64 %30, i64 %32, i64 2
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %25
%i648B

	full_text
	
i64 %28
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Nload8BD
B
	full_text5
3
1%44 = load double, double* %43, align 8, !tbaa !8
-double*8B

	full_text

double* %43
�getelementptr8B�
�
	full_text�
�
~%45 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %26, i64 %28, i64 %30, i64 %32, i64 2
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %26
%i648B

	full_text
	
i64 %28
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Nload8BD
B
	full_text5
3
1%46 = load double, double* %45, align 8, !tbaa !8
-double*8B

	full_text

double* %45
hcall8B^
\
	full_textO
M
K%47 = tail call double @llvm.fmuladd.f64(double %2, double %46, double %44)
+double8B

	full_text


double %46
+double8B

	full_text


double %44
Nstore8BC
A
	full_text4
2
0store double %47, double* %43, align 8, !tbaa !8
+double8B

	full_text


double %47
-double*8B

	full_text

double* %43
�getelementptr8B�
�
	full_text�
�
~%48 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %25, i64 %28, i64 %30, i64 %32, i64 3
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %25
%i648B

	full_text
	
i64 %28
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Nload8BD
B
	full_text5
3
1%49 = load double, double* %48, align 8, !tbaa !8
-double*8B

	full_text

double* %48
�getelementptr8B�
�
	full_text�
�
~%50 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %26, i64 %28, i64 %30, i64 %32, i64 3
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %26
%i648B

	full_text
	
i64 %28
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Nload8BD
B
	full_text5
3
1%51 = load double, double* %50, align 8, !tbaa !8
-double*8B

	full_text

double* %50
hcall8B^
\
	full_textO
M
K%52 = tail call double @llvm.fmuladd.f64(double %2, double %51, double %49)
+double8B

	full_text


double %51
+double8B

	full_text


double %49
Nstore8BC
A
	full_text4
2
0store double %52, double* %48, align 8, !tbaa !8
+double8B

	full_text


double %52
-double*8B

	full_text

double* %48
�getelementptr8B�
�
	full_text�
�
~%53 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %25, i64 %28, i64 %30, i64 %32, i64 4
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %25
%i648B

	full_text
	
i64 %28
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Nload8BD
B
	full_text5
3
1%54 = load double, double* %53, align 8, !tbaa !8
-double*8B

	full_text

double* %53
�getelementptr8B�
�
	full_text�
�
~%55 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %26, i64 %28, i64 %30, i64 %32, i64 4
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %26
%i648B

	full_text
	
i64 %28
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Nload8BD
B
	full_text5
3
1%56 = load double, double* %55, align 8, !tbaa !8
-double*8B

	full_text

double* %55
hcall8B^
\
	full_textO
M
K%57 = tail call double @llvm.fmuladd.f64(double %2, double %56, double %54)
+double8B

	full_text


double %56
+double8B

	full_text


double %54
Nstore8BC
A
	full_text4
2
0store double %57, double* %53, align 8, !tbaa !8
+double8B

	full_text


double %57
-double*8B

	full_text

double* %53
'br8B

	full_text

br label %58
$ret8B

	full_text


ret void
,double*8B

	full_text


double* %1
$i328B

	full_text


i32 %3
,double*8B

	full_text


double* %0
$i328B

	full_text


i32 %5
*double8B

	full_text

	double %2
$i328B

	full_text


i32 %4
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
#i328B

	full_text	

i32 1
$i648B

	full_text


i64 32
#i648B

	full_text	

i64 3
#i648B

	full_text	

i64 1
#i648B

	full_text	

i64 0
#i328B

	full_text	

i32 0
#i328B

	full_text	

i32 2
#i648B

	full_text	

i64 2
#i648B

	full_text	

i64 4
$i328B

	full_text


i32 -1        		 
 

                      !" !# $$ %& %% '( '' )* )) +, ++ -. -- /0 // 12 13 14 15 11 67 66 89 8: 8; 8< 88 => == ?@ ?A ?? BC BD BB EF EG EH EI EE JK JJ LM LN LO LP LL QR QQ ST SU SS VW VX VV YZ Y[ Y\ Y] YY ^_ ^^ `a `b `c `d `` ef ee gh gi gg jk jl jj mn mo mp mq mm rs rr tu tv tw tx tt yz yy {| {} {{ ~ ~	� ~~ �� �
� �
� �
� �� �� �� �� �
� �
� �
� �� �� �� �
� �
� �� �� �
� �� �� $� � #� � ?� S� g� {� ��    	    
          " &% ( *) ,
 .- 0# 2' 3+ 4/ 51 7$ 9' :+ ;/ <8 >= @6 A? C1 D# F' G+ H/ IE K$ M' N+ O/ PL RQ TJ US WE X# Z' [+ \/ ]Y _$ a' b+ c/ d` fe h^ ig kY l# n' o+ p/ qm s$ u' v+ w/ xt zy |r }{ m �# �' �+ �/ �� �$ �' �+ �/ �� �� �� �� �� �  �! #! �� � � �� �� �� � �� �	 �� 	g �� g? �� ? �� S �� S{ �� {� 	� %	� '	� )	� +	� -	� /	� m	� t	� 	� 	� 
	� E	� L	� 1	� 8� 	� 	� Y	� `
� �
� �	� 	� 	� "
ssor3"
_Z13get_global_idj"
llvm.fmuladd.f64*�
npb-LU-ssor3.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02�

wgsize_log1p
�}�A

transfer_bytes
���Z
 
transfer_bytes_log1p
�}�A

wgsize
 

devmap_label
