

[external]
KcallBC
A
	full_text4
2
0%5 = tail call i64 @_Z13get_global_idj(i32 1) #3
4truncB+
)
	full_text

%6 = trunc i64 %5 to i32
"i64B

	full_text


i64 %5
KcallBC
A
	full_text4
2
0%7 = tail call i64 @_Z13get_global_idj(i32 0) #3
3icmpB+
)
	full_text

%8 = icmp slt i32 %6, %3
"i32B

	full_text


i32 %6
4truncB+
)
	full_text

%9 = trunc i64 %7 to i32
"i64B

	full_text


i64 %7
4icmpB,
*
	full_text

%10 = icmp slt i32 %9, %2
"i32B

	full_text


i32 %9
.andB'
%
	full_text

%11 = and i1 %8, %10
 i1B

	full_text	

i1 %8
!i1B

	full_text


i1 %10
3icmpB+
)
	full_text

%12 = icmp sgt i32 %1, 0
/andB(
&
	full_text

%13 = and i1 %11, %12
!i1B

	full_text


i1 %11
!i1B

	full_text


i1 %12
8brB2
0
	full_text#
!
br i1 %13, label %14, label %31
!i1B

	full_text


i1 %13
0shl8B'
%
	full_text

%15 = shl i64 %5, 32
$i648B

	full_text


i64 %5
9ashr8B/
-
	full_text 

%16 = ashr exact i64 %15, 32
%i648B

	full_text
	
i64 %15
0shl8B'
%
	full_text

%17 = shl i64 %7, 32
$i648B

	full_text


i64 %7
9ashr8B/
-
	full_text 

%18 = ashr exact i64 %17, 32
%i648B

	full_text
	
i64 %17
6mul8B-
+
	full_text

%19 = mul nsw i64 %16, 845
%i648B

	full_text
	
i64 %16
5mul8B,
*
	full_text

%20 = mul nsw i64 %18, 65
%i648B

	full_text
	
i64 %18
2add8B)
'
	full_text

%21 = add i64 %19, %20
%i648B

	full_text
	
i64 %19
%i648B

	full_text
	
i64 %20
5zext8B+
)
	full_text

%22 = zext i32 %1 to i64
'br8B

	full_text

br label %23
Bphi8B9
7
	full_text*
(
&%24 = phi i64 [ 0, %14 ], [ %29, %23 ]
%i648B

	full_text
	
i64 %29
8mul8B/
-
	full_text 

%25 = mul nuw nsw i64 %24, 5
%i648B

	full_text
	
i64 %24
6add8B-
+
	full_text

%26 = add nsw i64 %21, %25
%i648B

	full_text
	
i64 %21
%i648B

	full_text
	
i64 %25
Ugetelementptr8BB
@
	full_text3
1
/%27 = getelementptr double, double* %0, i64 %26
%i648B

	full_text
	
i64 %26
@bitcast8B3
1
	full_text$
"
 %28 = bitcast double* %27 to i8*
-double*8B

	full_text

double* %27
?call8Bw
u
	full_texth
f
dcall void @memset_pattern16(i8* %28, i8* bitcast ([2 x double]* @.memset_pattern to i8*), i64 40) #4
%i8*8B

	full_text
	
i8* %28
8add8B/
-
	full_text 

%29 = add nuw nsw i64 %24, 1
%i648B

	full_text
	
i64 %24
7icmp8B-
+
	full_text

%30 = icmp eq i64 %29, %22
%i648B

	full_text
	
i64 %29
%i648B

	full_text
	
i64 %22
:br8B2
0
	full_text#
!
br i1 %30, label %31, label %23
#i18B

	full_text


i1 %30
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %1
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
,double*8B
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
$i648B

	full_text


i64 32
#i328B

	full_text	

i32 1
Qi8*8BF
D
	full_text7
5
3i8* bitcast ([2 x double]* @.memset_pattern to i8*)
$i648B

	full_text


i64 65
#i648B

	full_text	

i64 5
#i328B

	full_text	

i32 0
$i648B

	full_text


i64 40
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
%i648B

	full_text
	
i64 845        	
 		                      !  "    ## $& %% '( '' )* )+ )) ,- ,, ./ .. 01 00 23 22 45 46 44 78 7: : #; < 	= ,    
 	           ! "2 &% (  *' +) -, /. 1% 32 5# 64 8  9$ %7 97 % >> ?? 9 >>  >> 0 ?? 0@ @ @ @ A B 0C D 'E E F 0G 2H %I "
initialize1"
_Z13get_global_idj"
memset_pattern16*?
npb-BT-initialize1.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282

wgsize
<

wgsize_log1p
?fA

devmap_label
 
 
transfer_bytes_log1p
?fA

transfer_bytes
??n