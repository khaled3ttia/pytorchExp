

[external]
KcallBC
A
	full_text4
2
0%5 = tail call i64 @_Z13get_global_idj(i32 2) #2
,addB%
#
	full_text

%6 = add i64 %5, 1
"i64B

	full_text


i64 %5
4truncB+
)
	full_text

%7 = trunc i64 %6 to i32
"i64B

	full_text


i64 %6
KcallBC
A
	full_text4
2
0%8 = tail call i64 @_Z13get_global_idj(i32 1) #2
,addB%
#
	full_text

%9 = add i64 %8, 1
"i64B

	full_text


i64 %8
LcallBD
B
	full_text5
3
1%10 = tail call i64 @_Z13get_global_idj(i32 0) #2
.addB'
%
	full_text

%11 = add i64 %10, 1
#i64B

	full_text
	
i64 %10
2addB+
)
	full_text

%12 = add nsw i32 %3, -2
5icmpB-
+
	full_text

%13 = icmp slt i32 %12, %7
#i32B

	full_text
	
i32 %12
"i32B

	full_text


i32 %7
8brB2
0
	full_text#
!
br i1 %13, label %45, label %14
!i1B

	full_text


i1 %13
8trunc8B-
+
	full_text

%15 = trunc i64 %11 to i32
%i648B

	full_text
	
i64 %11
7trunc8B,
*
	full_text

%16 = trunc i64 %9 to i32
$i648B

	full_text


i64 %9
4add8B+
)
	full_text

%17 = add nsw i32 %2, -2
8icmp8B.
,
	full_text

%18 = icmp slt i32 %17, %16
%i328B

	full_text
	
i32 %17
%i328B

	full_text
	
i32 %16
4add8B+
)
	full_text

%19 = add nsw i32 %1, -2
8icmp8B.
,
	full_text

%20 = icmp slt i32 %19, %15
%i328B

	full_text
	
i32 %19
%i328B

	full_text
	
i32 %15
/or8B'
%
	full_text

%21 = or i1 %18, %20
#i18B

	full_text


i1 %18
#i18B

	full_text


i1 %20
:br8B2
0
	full_text#
!
br i1 %21, label %45, label %22
#i18B

	full_text


i1 %21
Wbitcast8BJ
H
	full_text;
9
7%23 = bitcast double* %0 to [65 x [65 x [5 x double]]]*
0shl8B'
%
	full_text

%24 = shl i64 %6, 32
$i648B

	full_text


i64 %6
9ashr8B/
-
	full_text 

%25 = ashr exact i64 %24, 32
%i648B

	full_text
	
i64 %24
0shl8B'
%
	full_text

%26 = shl i64 %9, 32
$i648B

	full_text


i64 %9
9ashr8B/
-
	full_text 

%27 = ashr exact i64 %26, 32
%i648B

	full_text
	
i64 %26
1shl8B(
&
	full_text

%28 = shl i64 %11, 32
%i648B

	full_text
	
i64 %11
9ashr8B/
-
	full_text 

%29 = ashr exact i64 %28, 32
%i648B

	full_text
	
i64 %28
?getelementptr8B?
?
	full_text~
|
z%30 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %23, i64 %25, i64 %27, i64 %29, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %23
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
Nload8BD
B
	full_text5
3
1%31 = load double, double* %30, align 8, !tbaa !8
-double*8B

	full_text

double* %30
@fmul8B6
4
	full_text'
%
#%32 = fmul double %31, 8.000000e-04
+double8B

	full_text


double %31
Nstore8BC
A
	full_text4
2
0store double %32, double* %30, align 8, !tbaa !8
+double8B

	full_text


double %32
-double*8B

	full_text

double* %30
?getelementptr8B?
?
	full_text~
|
z%33 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %23, i64 %25, i64 %27, i64 %29, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %23
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
Nload8BD
B
	full_text5
3
1%34 = load double, double* %33, align 8, !tbaa !8
-double*8B

	full_text

double* %33
@fmul8B6
4
	full_text'
%
#%35 = fmul double %34, 8.000000e-04
+double8B

	full_text


double %34
Nstore8BC
A
	full_text4
2
0store double %35, double* %33, align 8, !tbaa !8
+double8B

	full_text


double %35
-double*8B

	full_text

double* %33
?getelementptr8B?
?
	full_text~
|
z%36 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %23, i64 %25, i64 %27, i64 %29, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %23
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
Nload8BD
B
	full_text5
3
1%37 = load double, double* %36, align 8, !tbaa !8
-double*8B

	full_text

double* %36
@fmul8B6
4
	full_text'
%
#%38 = fmul double %37, 8.000000e-04
+double8B

	full_text


double %37
Nstore8BC
A
	full_text4
2
0store double %38, double* %36, align 8, !tbaa !8
+double8B

	full_text


double %38
-double*8B

	full_text

double* %36
?getelementptr8B?
?
	full_text~
|
z%39 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %23, i64 %25, i64 %27, i64 %29, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %23
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
Nload8BD
B
	full_text5
3
1%40 = load double, double* %39, align 8, !tbaa !8
-double*8B

	full_text

double* %39
@fmul8B6
4
	full_text'
%
#%41 = fmul double %40, 8.000000e-04
+double8B

	full_text


double %40
Nstore8BC
A
	full_text4
2
0store double %41, double* %39, align 8, !tbaa !8
+double8B

	full_text


double %41
-double*8B

	full_text

double* %39
?getelementptr8B?
?
	full_text~
|
z%42 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %23, i64 %25, i64 %27, i64 %29, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %23
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
Nload8BD
B
	full_text5
3
1%43 = load double, double* %42, align 8, !tbaa !8
-double*8B

	full_text

double* %42
@fmul8B6
4
	full_text'
%
#%44 = fmul double %43, 8.000000e-04
+double8B

	full_text


double %43
Nstore8BC
A
	full_text4
2
0store double %44, double* %42, align 8, !tbaa !8
+double8B

	full_text


double %44
-double*8B

	full_text

double* %42
'br8B

	full_text

br label %45
$ret8B

	full_text


ret void
,double*8B

	full_text


double* %0
$i328B

	full_text


i32 %2
$i328B

	full_text


i32 %3
$i328B

	full_text


i32 %1
-; undefined function B

	full_text

 
#i328B

	full_text	

i32 1
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
i32 -2
4double8B&
$
	full_text

double 8.000000e-04
#i648B

	full_text	

i64 4
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
i64 3
$i648B

	full_text


i64 32
#i648B

	full_text	

i64 1        		 
 

                      !" !# $% $$ &' && () (( *+ ** ,- ,, ./ .. 01 02 03 04 00 56 55 78 77 9: 9; 99 <= <> <? <@ << AB AA CD CC EF EG EE HI HJ HK HL HH MN MM OP OO QR QS QQ TU TV TW TX TT YZ YY [\ [[ ]^ ]_ ]] `a `b `c `d `` ef ee gh gg ij ik ii ln #o p q    	    
          " %$ ' )( +
 -, /# 1& 2* 3. 40 65 87 :0 ;# =& >* ?. @< BA DC F< G# I& J* K. LH NM PO RH S# U& V* W. XT ZY \[ ^T _# a& b* c. d` fe hg j` k m ! m! #l m m rr	 rr 	 rr  rr s t 0u Hv v v w 7w Cw Ow [w gx `y 	z { T| $| &| (| *| ,| .} } } 
} <"
compute_rhs6"
_Z13get_global_idj*?
npb-BT-compute_rhs6_A.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282?

wgsize_log1p
???A
 
transfer_bytes_log1p
???A

transfer_bytes	
????

wgsize
>

devmap_label
 