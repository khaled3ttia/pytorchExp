

[external]
KcallBC
A
	full_text4
2
0%6 = tail call i64 @_Z13get_global_idj(i32 2) #2
,addB%
#
	full_text

%7 = add i64 %6, 1
"i64B

	full_text


i64 %6
4truncB+
)
	full_text

%8 = trunc i64 %7 to i32
"i64B

	full_text


i64 %7
KcallBC
A
	full_text4
2
0%9 = tail call i64 @_Z13get_global_idj(i32 1) #2
-addB&
$
	full_text

%10 = add i64 %9, 1
"i64B

	full_text


i64 %9
LcallBD
B
	full_text5
3
1%11 = tail call i64 @_Z13get_global_idj(i32 0) #2
.addB'
%
	full_text

%12 = add i64 %11, 1
#i64B

	full_text
	
i64 %11
2addB+
)
	full_text

%13 = add nsw i32 %4, -1
5icmpB-
+
	full_text

%14 = icmp sgt i32 %13, %8
#i32B

	full_text
	
i32 %13
"i32B

	full_text


i32 %8
8brB2
0
	full_text#
!
br i1 %14, label %15, label %46
!i1B

	full_text


i1 %14
8trunc8B-
+
	full_text

%16 = trunc i64 %12 to i32
%i648B

	full_text
	
i64 %12
8trunc8B-
+
	full_text

%17 = trunc i64 %10 to i32
%i648B

	full_text
	
i64 %10
4add8B+
)
	full_text

%18 = add nsw i32 %3, -1
8icmp8B.
,
	full_text

%19 = icmp sgt i32 %18, %17
%i328B

	full_text
	
i32 %18
%i328B

	full_text
	
i32 %17
4add8B+
)
	full_text

%20 = add nsw i32 %2, -1
8icmp8B.
,
	full_text

%21 = icmp sgt i32 %20, %16
%i328B

	full_text
	
i32 %20
%i328B

	full_text
	
i32 %16
1and8B(
&
	full_text

%22 = and i1 %19, %21
#i18B

	full_text


i1 %19
#i18B

	full_text


i1 %21
:br8B2
0
	full_text#
!
br i1 %22, label %23, label %46
#i18B

	full_text


i1 %22
Ybitcast8BL
J
	full_text=
;
9%24 = bitcast double* %0 to [103 x [103 x [5 x double]]]*
0shl8B'
%
	full_text

%25 = shl i64 %7, 32
$i648B

	full_text


i64 %7
9ashr8B/
-
	full_text 

%26 = ashr exact i64 %25, 32
%i648B

	full_text
	
i64 %25
1shl8B(
&
	full_text

%27 = shl i64 %10, 32
%i648B

	full_text
	
i64 %10
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
%29 = shl i64 %12, 32
%i648B

	full_text
	
i64 %12
9ashr8B/
-
	full_text 

%30 = ashr exact i64 %29, 32
%i648B

	full_text
	
i64 %29
?getelementptr8B?
?
	full_text?
?
~%31 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %24, i64 %26, i64 %28, i64 %30, i64 0
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %24
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %28
%i648B

	full_text
	
i64 %30
Nload8BD
B
	full_text5
3
1%32 = load double, double* %31, align 8, !tbaa !8
-double*8B

	full_text

double* %31
6fmul8B,
*
	full_text

%33 = fmul double %32, %1
+double8B

	full_text


double %32
Nstore8BC
A
	full_text4
2
0store double %33, double* %31, align 8, !tbaa !8
+double8B

	full_text


double %33
-double*8B

	full_text

double* %31
?getelementptr8B?
?
	full_text?
?
~%34 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %24, i64 %26, i64 %28, i64 %30, i64 1
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %24
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %28
%i648B

	full_text
	
i64 %30
Nload8BD
B
	full_text5
3
1%35 = load double, double* %34, align 8, !tbaa !8
-double*8B

	full_text

double* %34
6fmul8B,
*
	full_text

%36 = fmul double %35, %1
+double8B

	full_text


double %35
Nstore8BC
A
	full_text4
2
0store double %36, double* %34, align 8, !tbaa !8
+double8B

	full_text


double %36
-double*8B

	full_text

double* %34
?getelementptr8B?
?
	full_text?
?
~%37 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %24, i64 %26, i64 %28, i64 %30, i64 2
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %24
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %28
%i648B

	full_text
	
i64 %30
Nload8BD
B
	full_text5
3
1%38 = load double, double* %37, align 8, !tbaa !8
-double*8B

	full_text

double* %37
6fmul8B,
*
	full_text

%39 = fmul double %38, %1
+double8B

	full_text


double %38
Nstore8BC
A
	full_text4
2
0store double %39, double* %37, align 8, !tbaa !8
+double8B

	full_text


double %39
-double*8B

	full_text

double* %37
?getelementptr8B?
?
	full_text?
?
~%40 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %24, i64 %26, i64 %28, i64 %30, i64 3
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %24
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %28
%i648B

	full_text
	
i64 %30
Nload8BD
B
	full_text5
3
1%41 = load double, double* %40, align 8, !tbaa !8
-double*8B

	full_text

double* %40
6fmul8B,
*
	full_text

%42 = fmul double %41, %1
+double8B

	full_text


double %41
Nstore8BC
A
	full_text4
2
0store double %42, double* %40, align 8, !tbaa !8
+double8B

	full_text


double %42
-double*8B

	full_text

double* %40
?getelementptr8B?
?
	full_text?
?
~%43 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %24, i64 %26, i64 %28, i64 %30, i64 4
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %24
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %28
%i648B

	full_text
	
i64 %30
Nload8BD
B
	full_text5
3
1%44 = load double, double* %43, align 8, !tbaa !8
-double*8B

	full_text

double* %43
6fmul8B,
*
	full_text

%45 = fmul double %44, %1
+double8B

	full_text


double %44
Nstore8BC
A
	full_text4
2
0store double %45, double* %43, align 8, !tbaa !8
+double8B

	full_text


double %45
-double*8B

	full_text

double* %43
'br8B

	full_text

br label %46
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %4
,double*8B

	full_text


double* %0
*double8B

	full_text

	double %1
$i328B

	full_text


i32 %3
$i328B

	full_text


i32 %2
-; undefined function B

	full_text

 
$i648B

	full_text


i64 32
#i328B

	full_text	

i32 2
#i328B

	full_text	

i32 1
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 4
#i648B

	full_text	

i64 1
#i648B

	full_text	

i64 3
#i648B

	full_text	

i64 0
#i648B

	full_text	

i64 2
$i328B

	full_text


i32 -1        		 
 

                      !" !# $% $$ &' && () (( *+ ** ,- ,, ./ .. 01 02 03 04 00 56 55 78 77 9: 9; 99 <= <> <? <@ << AB AA CD CC EF EG EE HI HJ HK HL HH MN MM OP OO QR QS QQ TU TV TW TX TT YZ YY [\ [[ ]^ ]_ ]] `a `b `c `d `` ef ee gh gg ij ik ii ln o #p 7p Cp Op [p gq r    	    
          " %$ ' )( +
 -, /# 1& 2* 3. 40 65 87 :0 ;# =& >* ?. @< BA DC F< G# I& J* K. LH NM PO RH S# U& V* W. XT ZY \[ ^T _# a& b* c. d` fe hg j` k  m! #! ml m ss m ss  ss 	 ss 	t $t &t (t *t ,t .u v w 	x `y y y 
y <z T{ 0| H} } } "
ssor2"
_Z13get_global_idj*?
npb-LU-ssor2.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282?

transfer_bytes
???Z

devmap_label


wgsize
 

wgsize_log1p
?}?A
 
transfer_bytes_log1p
?}?A