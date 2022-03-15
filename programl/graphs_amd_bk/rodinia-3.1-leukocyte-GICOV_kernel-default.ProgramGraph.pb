

[external]
LcallBD
B
	full_text5
3
1%11 = tail call i64 @_Z13get_global_idj(i32 0) #3
6truncB-
+
	full_text

%12 = trunc i64 %11 to i32
#i64B

	full_text
	
i64 %11
2mulB+
)
	full_text

%13 = mul nsw i32 %9, %8
6icmpB.
,
	full_text

%14 = icmp sgt i32 %13, %12
#i32B

	full_text
	
i32 %13
#i32B

	full_text
	
i32 %12
8brB2
0
	full_text#
!
br i1 %14, label %15, label %73
!i1B

	full_text


i1 %14
3sdiv8B)
'
	full_text

%16 = sdiv i32 %12, %8
%i328B

	full_text
	
i32 %12
5add8B,
*
	full_text

%17 = add nsw i32 %16, 22
%i328B

	full_text
	
i32 %16
3srem8B)
'
	full_text

%18 = srem i32 %12, %8
%i328B

	full_text
	
i32 %12
5add8B,
*
	full_text

%19 = add nsw i32 %18, 22
%i328B

	full_text
	
i32 %18
'br8B

	full_text

br label %20
Bphi8B9
7
	full_text*
(
&%21 = phi i64 [ 0, %15 ], [ %66, %59 ]
%i648B

	full_text
	
i64 %66
Ophi8BF
D
	full_text7
5
3%22 = phi float [ 0.000000e+00, %15 ], [ %65, %59 ]
)float8B

	full_text

	float %65
:mul8B1
/
	full_text"
 
%23 = mul nuw nsw i64 %21, 150
%i648B

	full_text
	
i64 %21
'br8B

	full_text

br label %24
Bphi8B9
7
	full_text*
(
&%25 = phi i64 [ 0, %20 ], [ %51, %24 ]
%i648B

	full_text
	
i64 %51
Ophi8BF
D
	full_text7
5
3%26 = phi float [ 0.000000e+00, %20 ], [ %55, %24 ]
)float8B

	full_text

	float %55
Ophi8BF
D
	full_text7
5
3%27 = phi float [ 0.000000e+00, %20 ], [ %57, %24 ]
)float8B

	full_text

	float %57
Ophi8BF
D
	full_text7
5
3%28 = phi float [ 0.000000e+00, %20 ], [ %49, %24 ]
)float8B

	full_text

	float %49
:add8B1
/
	full_text"
 
%29 = add nuw nsw i64 %25, %23
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %23
Xgetelementptr8BE
C
	full_text6
4
2%30 = getelementptr inbounds i32, i32* %6, i64 %29
%i648B

	full_text
	
i64 %29
Hload8B>
<
	full_text/
-
+%31 = load i32, i32* %30, align 4, !tbaa !8
'i32*8B

	full_text


i32* %30
6add8B-
+
	full_text

%32 = add nsw i32 %31, %19
%i328B

	full_text
	
i32 %31
%i328B

	full_text
	
i32 %19
Xgetelementptr8BE
C
	full_text6
4
2%33 = getelementptr inbounds i32, i32* %5, i64 %29
%i648B

	full_text
	
i64 %29
Hload8B>
<
	full_text/
-
+%34 = load i32, i32* %33, align 4, !tbaa !8
'i32*8B

	full_text


i32* %33
6add8B-
+
	full_text

%35 = add nsw i32 %34, %17
%i328B

	full_text
	
i32 %34
%i328B

	full_text
	
i32 %17
5mul8B,
*
	full_text

%36 = mul nsw i32 %35, %0
%i328B

	full_text
	
i32 %35
6add8B-
+
	full_text

%37 = add nsw i32 %32, %36
%i328B

	full_text
	
i32 %32
%i328B

	full_text
	
i32 %36
6sext8B,
*
	full_text

%38 = sext i32 %37 to i64
%i328B

	full_text
	
i32 %37
\getelementptr8BI
G
	full_text:
8
6%39 = getelementptr inbounds float, float* %1, i64 %38
%i648B

	full_text
	
i64 %38
Mload8BC
A
	full_text4
2
0%40 = load float, float* %39, align 4, !tbaa !12
+float*8B

	full_text


float* %39
\getelementptr8BI
G
	full_text:
8
6%41 = getelementptr inbounds float, float* %4, i64 %25
%i648B

	full_text
	
i64 %25
Mload8BC
A
	full_text4
2
0%42 = load float, float* %41, align 4, !tbaa !12
+float*8B

	full_text


float* %41
\getelementptr8BI
G
	full_text:
8
6%43 = getelementptr inbounds float, float* %2, i64 %38
%i648B

	full_text
	
i64 %38
Mload8BC
A
	full_text4
2
0%44 = load float, float* %43, align 4, !tbaa !12
+float*8B

	full_text


float* %43
\getelementptr8BI
G
	full_text:
8
6%45 = getelementptr inbounds float, float* %3, i64 %25
%i648B

	full_text
	
i64 %25
Mload8BC
A
	full_text4
2
0%46 = load float, float* %45, align 4, !tbaa !12
+float*8B

	full_text


float* %45
6fmul8B,
*
	full_text

%47 = fmul float %44, %46
)float8B

	full_text

	float %44
)float8B

	full_text

	float %46
ecall8B[
Y
	full_textL
J
H%48 = tail call float @llvm.fmuladd.f32(float %40, float %42, float %47)
)float8B

	full_text

	float %40
)float8B

	full_text

	float %42
)float8B

	full_text

	float %47
6fadd8B,
*
	full_text

%49 = fadd float %28, %48
)float8B

	full_text

	float %28
)float8B

	full_text

	float %48
6fsub8B,
*
	full_text

%50 = fsub float %48, %26
)float8B

	full_text

	float %48
)float8B

	full_text

	float %26
8add8B/
-
	full_text 

%51 = add nuw nsw i64 %25, 1
%i648B

	full_text
	
i64 %25
8trunc8B-
+
	full_text

%52 = trunc i64 %51 to i32
%i648B

	full_text
	
i64 %51
<sitofp8B0
.
	full_text!

%53 = sitofp i32 %52 to float
%i328B

	full_text
	
i32 %52
Cfdiv8B9
7
	full_text*
(
&%54 = fdiv float %50, %53, !fpmath !14
)float8B

	full_text

	float %50
)float8B

	full_text

	float %53
6fadd8B,
*
	full_text

%55 = fadd float %26, %54
)float8B

	full_text

	float %26
)float8B

	full_text

	float %54
6fsub8B,
*
	full_text

%56 = fsub float %48, %55
)float8B

	full_text

	float %48
)float8B

	full_text

	float %55
ecall8B[
Y
	full_textL
J
H%57 = tail call float @llvm.fmuladd.f32(float %50, float %56, float %27)
)float8B

	full_text

	float %50
)float8B

	full_text

	float %56
)float8B

	full_text

	float %27
7icmp8B-
+
	full_text

%58 = icmp eq i64 %51, 150
%i648B

	full_text
	
i64 %51
:br8B2
0
	full_text#
!
br i1 %58, label %59, label %24
#i18B

	full_text


i1 %58
Lfdiv8BB
@
	full_text3
1
/%60 = fdiv float %49, 1.500000e+02, !fpmath !14
)float8B

	full_text

	float %49
Lfdiv8BB
@
	full_text3
1
/%61 = fdiv float %57, 1.490000e+02, !fpmath !14
)float8B

	full_text

	float %57
6fmul8B,
*
	full_text

%62 = fmul float %60, %60
)float8B

	full_text

	float %60
)float8B

	full_text

	float %60
Cfdiv8B9
7
	full_text*
(
&%63 = fdiv float %62, %61, !fpmath !14
)float8B

	full_text

	float %62
)float8B

	full_text

	float %61
:fcmp8B0
.
	full_text!

%64 = fcmp ogt float %63, %22
)float8B

	full_text

	float %63
)float8B

	full_text

	float %22
Hselect8B<
:
	full_text-
+
)%65 = select i1 %64, float %63, float %22
#i18B

	full_text


i1 %64
)float8B

	full_text

	float %63
)float8B

	full_text

	float %22
8add8B/
-
	full_text 

%66 = add nuw nsw i64 %21, 1
%i648B

	full_text
	
i64 %21
5icmp8B+
)
	full_text

%67 = icmp eq i64 %66, 7
%i648B

	full_text
	
i64 %66
:br8B2
0
	full_text#
!
br i1 %67, label %68, label %20
#i18B

	full_text


i1 %67
5mul8B,
*
	full_text

%69 = mul nsw i32 %17, %0
%i328B

	full_text
	
i32 %17
6add8B-
+
	full_text

%70 = add nsw i32 %69, %19
%i328B

	full_text
	
i32 %69
%i328B

	full_text
	
i32 %19
6sext8B,
*
	full_text

%71 = sext i32 %70 to i64
%i328B

	full_text
	
i32 %70
\getelementptr8BI
G
	full_text:
8
6%72 = getelementptr inbounds float, float* %7, i64 %71
%i648B

	full_text
	
i64 %71
Mstore8BB
@
	full_text3
1
/store float %65, float* %72, align 4, !tbaa !12
)float8B

	full_text

	float %65
+float*8B

	full_text


float* %72
'br8B

	full_text

br label %73
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %0
$i328B

	full_text


i32 %8
$i328B

	full_text


i32 %9
&i32*8B

	full_text
	
i32* %5
*float*8B

	full_text

	float* %4
*float*8B

	full_text

	float* %3
*float*8B

	full_text

	float* %7
*float*8B

	full_text

	float* %1
*float*8B

	full_text

	float* %2
&i32*8B

	full_text
	
i32* %6
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
#i648B

	full_text	

i64 1
#i648B

	full_text	

i64 7
#i648B

	full_text	

i64 0
%i648B

	full_text
	
i64 150
#i328B

	full_text	

i32 0
2float8B%
#
	full_text

float 0.000000e+00
2float8B%
#
	full_text

float 1.500000e+02
2float8B%
#
	full_text

float 1.490000e+02
$i328B

	full_text


i32 22       	  

                    !    "# "$ "" %& %% '( '' )* )+ )) ,- ,, ./ .. 01 02 00 34 33 56 57 55 89 88 :; :: <= << >? >> @A @@ BC BB DE DD FG FF HI HH JK JL JJ MN MO MP MM QR QS QQ TU TV TT WX WW YZ YY [\ [[ ]^ ]_ ]] `a `b `` cd ce cc fg fh fi ff jk jj lm lo nn pq pp rs rt rr uv uw uu xy xz xx {| {} {~ {{ €  ‚  ƒ„ ƒ† …… ‡ˆ ‡
‰ ‡‡ Š‹ ŠŠ Œ
 ŒŒ Ž Ž
 ŽŽ ‘	“ 3
“ …	” 	” 
	” • – ,— >˜ F™ Œš :› Bœ %    	 
    {  W ` f Q ! # $" &% (' * +" -, /. 1 20 4) 63 75 98 ;: = ?> A8 CB E GF ID KH L< N@ OJ P  RM SM U V XW ZY \T ^[ _ a] bM d` eT gc h iW kj mQ of qn sn tr vp wu y zx |u } ~ € ‚ „ †… ˆ ‰‡ ‹Š { Œ  
 ’  l nl ƒ …ƒ ‘ ’  žž ’M žž Mf žž f  	Ÿ W	Ÿ 
  ¡ ¡ 	¢ 	¢ j£ ¤ ¤ ¤ ¤  	¥ n	¦ p	§ 	§ "
GICOV_kernel"
_Z13get_global_idj"
llvm.fmuladd.f32*ž
%rodinia-3.1-leukocyte-GICOV_kernel.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02

wgsize
€

devmap_label


transfer_bytes
„ÜÉ
 
transfer_bytes_log1p
G2‚A

wgsize_log1p
G2‚A