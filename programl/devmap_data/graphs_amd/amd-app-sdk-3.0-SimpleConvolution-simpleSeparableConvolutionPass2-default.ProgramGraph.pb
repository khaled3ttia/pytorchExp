

[external]
LextractelementB:
8
	full_text+
)
'%7 = extractelement <2 x i32> %3, i64 0
LextractelementB:
8
	full_text+
)
'%8 = extractelement <2 x i32> %3, i64 1
KcallBC
A
	full_text4
2
0%9 = tail call i64 @_Z13get_global_idj(i32 0) #2
5truncB,
*
	full_text

%10 = trunc i64 %9 to i32
"i64B

	full_text


i64 %9
1uremB)
'
	full_text

%11 = urem i32 %10, %8
#i32B

	full_text
	
i32 %10
"i32B

	full_text


i32 %8
1udivB)
'
	full_text

%12 = udiv i32 %10, %8
#i32B

	full_text
	
i32 %10
"i32B

	full_text


i32 %8
5icmpB-
+
	full_text

%13 = icmp ult i32 %12, %7
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
br i1 %13, label %14, label %46
!i1B

	full_text


i1 %13
1add8B(
&
	full_text

%15 = add i32 %11, %4
%i328B

	full_text
	
i32 %11
8icmp8B.
,
	full_text

%16 = icmp ult i32 %11, %15
%i328B

	full_text
	
i32 %11
%i328B

	full_text
	
i32 %15
:br8B2
0
	full_text#
!
br i1 %16, label %17, label %25
#i18B

	full_text


i1 %16
Oextractelement8B;
9
	full_text,
*
(%18 = extractelement <2 x i32> %5, i64 1
2mul8B)
'
	full_text

%19 = mul i32 %12, %18
%i328B

	full_text
	
i32 %12
%i328B

	full_text
	
i32 %18
6zext8B,
*
	full_text

%20 = zext i32 %11 to i64
%i328B

	full_text
	
i32 %11
5zext8B+
)
	full_text

%21 = zext i32 %4 to i64
'br8B

	full_text

br label %31
?fadd8B5
3
	full_text&
$
"%23 = fadd float %43, 5.000000e-01
)float8B

	full_text

	float %43
<fptosi8B0
.
	full_text!

%24 = fptosi float %23 to i32
)float8B

	full_text

	float %23
'br8B

	full_text

br label %25
Bphi8B9
7
	full_text*
(
&%26 = phi i32 [ 0, %14 ], [ %24, %22 ]
%i328B

	full_text
	
i32 %24
1mul8B(
&
	full_text

%27 = mul i32 %11, %7
%i328B

	full_text
	
i32 %11
$i328B

	full_text


i32 %7
2add8B)
'
	full_text

%28 = add i32 %27, %12
%i328B

	full_text
	
i32 %27
%i328B

	full_text
	
i32 %12
6zext8B,
*
	full_text

%29 = zext i32 %28 to i64
%i328B

	full_text
	
i32 %28
Xgetelementptr8BE
C
	full_text6
4
2%30 = getelementptr inbounds i32, i32* %2, i64 %29
%i648B

	full_text
	
i64 %29
Hstore8B=
;
	full_text.
,
*store i32 %26, i32* %30, align 4, !tbaa !9
%i328B

	full_text
	
i32 %26
'i32*8B

	full_text


i32* %30
'br8B

	full_text

br label %46
Bphi8B9
7
	full_text*
(
&%32 = phi i64 [ 0, %17 ], [ %40, %31 ]
%i648B

	full_text
	
i64 %40
Dphi8B;
9
	full_text,
*
(%33 = phi i64 [ %20, %17 ], [ %44, %31 ]
%i648B

	full_text
	
i64 %20
%i648B

	full_text
	
i64 %44
Ophi8BF
D
	full_text7
5
3%34 = phi float [ 0.000000e+00, %17 ], [ %43, %31 ]
)float8B

	full_text

	float %43
8trunc8B-
+
	full_text

%35 = trunc i64 %33 to i32
%i648B

	full_text
	
i64 %33
2add8B)
'
	full_text

%36 = add i32 %19, %35
%i328B

	full_text
	
i32 %19
%i328B

	full_text
	
i32 %35
6zext8B,
*
	full_text

%37 = zext i32 %36 to i64
%i328B

	full_text
	
i32 %36
\getelementptr8BI
G
	full_text:
8
6%38 = getelementptr inbounds float, float* %0, i64 %37
%i648B

	full_text
	
i64 %37
Mload8BC
A
	full_text4
2
0%39 = load float, float* %38, align 4, !tbaa !13
+float*8B

	full_text


float* %38
8add8B/
-
	full_text 

%40 = add nuw nsw i64 %32, 1
%i648B

	full_text
	
i64 %32
\getelementptr8BI
G
	full_text:
8
6%41 = getelementptr inbounds float, float* %1, i64 %32
%i648B

	full_text
	
i64 %32
Mload8BC
A
	full_text4
2
0%42 = load float, float* %41, align 4, !tbaa !13
+float*8B

	full_text


float* %41
acall8BW
U
	full_textH
F
D%43 = tail call float @_Z3madfff(float %39, float %42, float %34) #2
)float8B

	full_text

	float %39
)float8B

	full_text

	float %42
)float8B

	full_text

	float %34
8add8B/
-
	full_text 

%44 = add nuw nsw i64 %33, 1
%i648B

	full_text
	
i64 %33
7icmp8B-
+
	full_text

%45 = icmp eq i64 %40, %21
%i648B

	full_text
	
i64 %40
%i648B

	full_text
	
i64 %21
:br8B2
0
	full_text#
!
br i1 %45, label %22, label %31
#i18B

	full_text


i1 %45
$ret8B

	full_text


ret void
0	<2 x i32>8B

	full_text

<2 x i32> %5
*float*8B

	full_text

	float* %1
$i328B

	full_text


i32 %4
&i32*8B

	full_text
	
i32* %2
*float*8B

	full_text

	float* %0
0	<2 x i32>8B

	full_text

<2 x i32> %3
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
2float8B%
#
	full_text

float 0.000000e+00
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 0
#i648B

	full_text	

i64 1
2float8B%
#
	full_text

float 5.000000e-01        	
 	 		                  !    "# "" $& %% '( ') '' *+ *, ** -. -- /0 // 12 13 11 46 55 78 79 77 :; :: <= << >? >@ >> AB AA CD CC EF EE GH GG IJ II KL KK MN MO MP MM QR QQ ST SU SS VW VY Z I[ [ \ /] C^ ^     
 	       	   M !  #" & ( )' +	 ,* .- 0% 2/ 3G 6 8Q 9M ;7 = ?< @> BA DC F5 H5 JI LE NK O: P7 RG T US W  X  % 54 XV  V 5$ % X __ `` __ M `` Ma :b b %c c 5d d d Gd Qe  "!
simpleSeparableConvolutionPass2"
_Z13get_global_idj"
	_Z3madfff*?
4SimpleConvolution-simpleSeparableConvolutionPass2.clu
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

wgsize_log1p
?tA
 
transfer_bytes_log1p
?tA

wgsize
?

devmap_label
