

[external]
KcallBC
A
	full_text4
2
0%6 = tail call i64 @_Z13get_global_idj(i32 2) #3
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
0%8 = tail call i64 @_Z13get_global_idj(i32 1) #3
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
1%10 = tail call i64 @_Z13get_global_idj(i32 0) #3
6truncB-
+
	full_text

%11 = trunc i64 %10 to i32
#i64B

	full_text
	
i64 %10
4icmpB,
*
	full_text

%12 = icmp slt i32 %7, %4
"i32B

	full_text


i32 %7
4icmpB,
*
	full_text

%13 = icmp slt i32 %9, %3
"i32B

	full_text


i32 %9
/andB(
&
	full_text

%14 = and i1 %12, %13
!i1B

	full_text


i1 %12
!i1B

	full_text


i1 %13
8brB2
0
	full_text#
!
br i1 %14, label %15, label %29
!i1B

	full_text


i1 %14
Ybitcast8BL
J
	full_text=
;
9%16 = bitcast double* %0 to [103 x [103 x [5 x double]]]*
pcall8Bf
d
	full_textW
U
S%17 = tail call double @exact_scalar(i32 0, i32 %9, i32 %7, i32 %11, double* %1) #4
$i328B

	full_text


i32 %9
$i328B

	full_text


i32 %7
%i328B

	full_text
	
i32 %11
0shl8B'
%
	full_text

%18 = shl i64 %6, 32
$i648B

	full_text


i64 %6
9ashr8B/
-
	full_text 

%19 = ashr exact i64 %18, 32
%i648B

	full_text
	
i64 %18
0shl8B'
%
	full_text

%20 = shl i64 %8, 32
$i648B

	full_text


i64 %8
9ashr8B/
-
	full_text 

%21 = ashr exact i64 %20, 32
%i648B

	full_text
	
i64 %20
1shl8B(
&
	full_text

%22 = shl i64 %10, 32
%i648B

	full_text
	
i64 %10
9ashr8B/
-
	full_text 

%23 = ashr exact i64 %22, 32
%i648B

	full_text
	
i64 %22
?getelementptr8B?
?
	full_text?
?
~%24 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %16, i64 %19, i64 %21, i64 0, i64 %23
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %16
%i648B

	full_text
	
i64 %19
%i648B

	full_text
	
i64 %21
%i648B

	full_text
	
i64 %23
Nstore8BC
A
	full_text4
2
0store double %17, double* %24, align 8, !tbaa !8
+double8B

	full_text


double %17
-double*8B

	full_text

double* %24
4add8B+
)
	full_text

%25 = add nsw i32 %2, -1
rcall8Bh
f
	full_textY
W
U%26 = tail call double @exact_scalar(i32 %25, i32 %9, i32 %7, i32 %11, double* %1) #4
%i328B

	full_text
	
i32 %25
$i328B

	full_text


i32 %9
$i328B

	full_text


i32 %7
%i328B

	full_text
	
i32 %11
6sext8B,
*
	full_text

%27 = sext i32 %25 to i64
%i328B

	full_text
	
i32 %25
?getelementptr8B?
?
	full_text?
?
?%28 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %16, i64 %19, i64 %21, i64 %27, i64 %23
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %16
%i648B

	full_text
	
i64 %19
%i648B

	full_text
	
i64 %21
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %23
Nstore8BC
A
	full_text4
2
0store double %26, double* %28, align 8, !tbaa !8
+double8B

	full_text


double %26
-double*8B

	full_text

double* %28
'br8B

	full_text

br label %29
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %3
$i328B

	full_text


i32 %2
$i328B

	full_text


i32 %4
,double*8B

	full_text


double* %1
,double*8B

	full_text


double* %0
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
#i328B

	full_text	

i32 0
$i648B

	full_text


i64 32
#i328B

	full_text	

i32 2
$i328B

	full_text


i32 -1
#i648B

	full_text	

i64 0
#i328B

	full_text	

i32 1       	  
 

                     !    "# "" $% $& $' $( $$ )* )+ )) ,, -. -/ -0 -1 -- 23 22 45 46 47 48 49 44 :; :< :: =? @ ,A 
B B -C    	  
           !  # % & '" ( *$ +, . / 0 1, 3 5 6 72 8" 9- ;4 <  >= > > DD EE DD  DD  DD - EE - EE F F G G G G G  G "H I ,J $K "
setbv3"
_Z13get_global_idj"
exact_scalar*?
npb-LU-setbv3.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02?

wgsize

 
transfer_bytes_log1p
?}?A

wgsize_log1p
?}?A

transfer_bytes
Ȑ?Z

devmap_label
 