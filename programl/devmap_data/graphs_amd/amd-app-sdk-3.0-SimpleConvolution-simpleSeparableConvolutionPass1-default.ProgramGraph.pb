

[external]
LextractelementB:
8
	full_text+
)
'%7 = extractelement <2 x i32> %3, i64 0
KcallBC
A
	full_text4
2
0%8 = tail call i64 @_Z13get_global_idj(i32 0) #2
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
0uremB(
&
	full_text

%10 = urem i32 %9, %7
"i32B

	full_text


i32 %9
"i32B

	full_text


i32 %7
0udivB(
&
	full_text

%11 = udiv i32 %9, %7
"i32B

	full_text


i32 %9
"i32B

	full_text


i32 %7
MextractelementB;
9
	full_text,
*
(%12 = extractelement <2 x i32> %3, i64 1
.addB'
%
	full_text

%13 = add i32 %4, -1
0addB)
'
	full_text

%14 = add i32 %13, %12
#i32B

	full_text
	
i32 %13
#i32B

	full_text
	
i32 %12
6icmpB.
,
	full_text

%15 = icmp ult i32 %11, %14
#i32B

	full_text
	
i32 %11
#i32B

	full_text
	
i32 %14
8brB2
0
	full_text#
!
br i1 %15, label %16, label %47
!i1B

	full_text


i1 %15
1add8B(
&
	full_text

%17 = add i32 %10, %4
%i328B

	full_text
	
i32 %10
8icmp8B.
,
	full_text

%18 = icmp ult i32 %10, %17
%i328B

	full_text
	
i32 %10
%i328B

	full_text
	
i32 %17
:br8B2
0
	full_text#
!
br i1 %18, label %19, label %24
#i18B

	full_text


i1 %18
Oextractelement8B;
9
	full_text,
*
(%20 = extractelement <2 x i32> %5, i64 0
2mul8B)
'
	full_text

%21 = mul i32 %11, %20
%i328B

	full_text
	
i32 %11
%i328B

	full_text
	
i32 %20
6zext8B,
*
	full_text

%22 = zext i32 %10 to i64
%i328B

	full_text
	
i32 %10
5zext8B+
)
	full_text

%23 = zext i32 %4 to i64
'br8B

	full_text

br label %31
Ophi8BF
D
	full_text7
5
3%25 = phi float [ 0.000000e+00, %16 ], [ %44, %31 ]
)float8B

	full_text

	float %44
Oextractelement8B;
9
	full_text,
*
(%26 = extractelement <2 x i32> %5, i64 1
2mul8B)
'
	full_text

%27 = mul i32 %10, %26
%i328B

	full_text
	
i32 %10
%i328B

	full_text
	
i32 %26
2add8B)
'
	full_text

%28 = add i32 %27, %11
%i328B

	full_text
	
i32 %27
%i328B

	full_text
	
i32 %11
6zext8B,
*
	full_text

%29 = zext i32 %28 to i64
%i328B

	full_text
	
i32 %28
\getelementptr8BI
G
	full_text:
8
6%30 = getelementptr inbounds float, float* %2, i64 %29
%i648B

	full_text
	
i64 %29
Lstore8BA
?
	full_text2
0
.store float %25, float* %30, align 4, !tbaa !9
)float8B

	full_text

	float %25
+float*8B

	full_text


float* %30
'br8B

	full_text

br label %47
Bphi8B9
7
	full_text*
(
&%32 = phi i64 [ 0, %19 ], [ %41, %31 ]
%i648B

	full_text
	
i64 %41
Dphi8B;
9
	full_text,
*
(%33 = phi i64 [ %22, %19 ], [ %45, %31 ]
%i648B

	full_text
	
i64 %22
%i648B

	full_text
	
i64 %45
Ophi8BF
D
	full_text7
5
3%34 = phi float [ 0.000000e+00, %19 ], [ %44, %31 ]
)float8B

	full_text

	float %44
8trunc8B-
+
	full_text

%35 = trunc i64 %33 to i32
%i648B

	full_text
	
i64 %33
2add8B)
'
	full_text

%36 = add i32 %21, %35
%i328B

	full_text
	
i32 %21
%i328B

	full_text
	
i32 %35
6zext8B,
*
	full_text

%37 = zext i32 %36 to i64
%i328B

	full_text
	
i32 %36
Xgetelementptr8BE
C
	full_text6
4
2%38 = getelementptr inbounds i32, i32* %0, i64 %37
%i648B

	full_text
	
i64 %37
Iload8B?
=
	full_text0
.
,%39 = load i32, i32* %38, align 4, !tbaa !13
'i32*8B

	full_text


i32* %38
<uitofp8B0
.
	full_text!

%40 = uitofp i32 %39 to float
%i328B

	full_text
	
i32 %39
8add8B/
-
	full_text 

%41 = add nuw nsw i64 %32, 1
%i648B

	full_text
	
i64 %32
\getelementptr8BI
G
	full_text:
8
6%42 = getelementptr inbounds float, float* %1, i64 %32
%i648B

	full_text
	
i64 %32
Lload8BB
@
	full_text3
1
/%43 = load float, float* %42, align 4, !tbaa !9
+float*8B

	full_text


float* %42
acall8BW
U
	full_textH
F
D%44 = tail call float @_Z3madfff(float %40, float %43, float %34) #2
)float8B

	full_text

	float %40
)float8B

	full_text

	float %43
)float8B

	full_text

	float %34
8add8B/
-
	full_text 

%45 = add nuw nsw i64 %33, 1
%i648B

	full_text
	
i64 %33
7icmp8B-
+
	full_text

%46 = icmp eq i64 %41, %23
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %23
:br8B2
0
	full_text#
!
br i1 %46, label %24, label %31
#i18B

	full_text


i1 %46
$ret8B

	full_text


ret void
*float*8B

	full_text

	float* %2
&i32*8B

	full_text
	
i32* %0
*float*8B

	full_text

	float* %1
$i328B

	full_text


i32 %4
0	<2 x i32>8B

	full_text

<2 x i32> %3
0	<2 x i32>8B

	full_text

<2 x i32> %5
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
#i328B

	full_text	

i32 0
$i328B

	full_text


i32 -1
2float8B%
#
	full_text

float 0.000000e+00
#i648B

	full_text	

i64 1
#i648B

	full_text	

i64 0       	 
                      !    "" #% $$ && '( ') '' *+ *, ** -. -- /0 // 12 13 11 46 55 78 79 77 :; :: <= << >? >@ >> AB AA CD CC EF EE GH GG IJ II KL KK MN MM OP OQ OR OO ST SS UV UW UU XY X[ /\ C] K^ ^ ^ "_ _ ` ` &    	 
            !O % (& )' + ,* .- 0$ 2/ 3I 6  8S 9O ;7 = ?< @> BA DC FE H5 J5 LK NG PM Q: R7 TI V" WU Y  Z  $# 54 ZX $X 5 Z aa bb aa O bb Oc d e $e :f f &f If Sg g g 5"!
simpleSeparableConvolutionPass1"
_Z13get_global_idj"
	_Z3madfff*?
4SimpleConvolution-simpleSeparableConvolutionPass1.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282?

transfer_bytes
???

devmap_label


wgsize_log1p
?tA
 
transfer_bytes_log1p
?tA

wgsize
?