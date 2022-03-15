

[external]
KcallBC
A
	full_text4
2
0%7 = tail call i64 @_Z13get_global_idj(i32 0) #3
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
3icmpB+
)
	full_text

%9 = icmp slt i32 %8, %4
"i32B

	full_text


i32 %8
7brB1
/
	full_text"
 
br i1 %9, label %10, label %72
 i1B

	full_text	

i1 %9
0shl8B'
%
	full_text

%11 = shl i64 %7, 32
$i648B

	full_text


i64 %7
9ashr8B/
-
	full_text 

%12 = ashr exact i64 %11, 32
%i648B

	full_text
	
i64 %11
Xgetelementptr8BE
C
	full_text6
4
2%13 = getelementptr inbounds i32, i32* %3, i64 %12
%i648B

	full_text
	
i64 %12
Hload8B>
<
	full_text/
-
+%14 = load i32, i32* %13, align 4, !tbaa !8
'i32*8B

	full_text


i32* %13
6icmp8B,
*
	full_text

%15 = icmp sgt i32 %14, 0
%i328B

	full_text
	
i32 %14
:br8B2
0
	full_text#
!
br i1 %15, label %16, label %41
#i18B

	full_text


i1 %15
5sext8B+
)
	full_text

%17 = sext i32 %4 to i64
0shl8B'
%
	full_text

%18 = shl i64 %7, 32
$i648B

	full_text


i64 %7
9ashr8B/
-
	full_text 

%19 = ashr exact i64 %18, 32
%i648B

	full_text
	
i64 %18
6zext8B,
*
	full_text

%20 = zext i32 %14 to i64
%i328B

	full_text
	
i32 %14
0and8B'
%
	full_text

%21 = and i64 %20, 1
%i648B

	full_text
	
i64 %20
5icmp8B+
)
	full_text

%22 = icmp eq i32 %14, 1
%i328B

	full_text
	
i32 %14
:br8B2
0
	full_text#
!
br i1 %22, label %25, label %23
#i18B

	full_text


i1 %22
6sub8B-
+
	full_text

%24 = sub nsw i64 %20, %21
%i648B

	full_text
	
i64 %20
%i648B

	full_text
	
i64 %21
'br8B

	full_text

br label %44
Hphi8B?
=
	full_text0
.
,%26 = phi float [ undef, %16 ], [ %68, %44 ]
)float8B

	full_text

	float %68
Bphi8B9
7
	full_text*
(
&%27 = phi i64 [ 0, %16 ], [ %69, %44 ]
%i648B

	full_text
	
i64 %69
Ophi8BF
D
	full_text7
5
3%28 = phi float [ 0.000000e+00, %16 ], [ %68, %44 ]
)float8B

	full_text

	float %68
5icmp8B+
)
	full_text

%29 = icmp eq i64 %21, 0
%i648B

	full_text
	
i64 %21
:br8B2
0
	full_text#
!
br i1 %29, label %41, label %30
#i18B

	full_text


i1 %29
6mul8B-
+
	full_text

%31 = mul nsw i64 %27, %17
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %17
6add8B-
+
	full_text

%32 = add nsw i64 %31, %19
%i648B

	full_text
	
i64 %31
%i648B

	full_text
	
i64 %19
\getelementptr8BI
G
	full_text:
8
6%33 = getelementptr inbounds float, float* %0, i64 %32
%i648B

	full_text
	
i64 %32
Mload8BC
A
	full_text4
2
0%34 = load float, float* %33, align 4, !tbaa !12
+float*8B

	full_text


float* %33
Xgetelementptr8BE
C
	full_text6
4
2%35 = getelementptr inbounds i32, i32* %2, i64 %32
%i648B

	full_text
	
i64 %32
Hload8B>
<
	full_text/
-
+%36 = load i32, i32* %35, align 4, !tbaa !8
'i32*8B

	full_text


i32* %35
6sext8B,
*
	full_text

%37 = sext i32 %36 to i64
%i328B

	full_text
	
i32 %36
\getelementptr8BI
G
	full_text:
8
6%38 = getelementptr inbounds float, float* %1, i64 %37
%i648B

	full_text
	
i64 %37
Mload8BC
A
	full_text4
2
0%39 = load float, float* %38, align 4, !tbaa !12
+float*8B

	full_text


float* %38
ecall8B[
Y
	full_textL
J
H%40 = tail call float @llvm.fmuladd.f32(float %34, float %39, float %28)
)float8B

	full_text

	float %34
)float8B

	full_text

	float %39
)float8B

	full_text

	float %28
'br8B

	full_text

br label %41
]phi8BT
R
	full_textE
C
A%42 = phi float [ 0.000000e+00, %10 ], [ %26, %25 ], [ %40, %30 ]
)float8B

	full_text

	float %26
)float8B

	full_text

	float %40
\getelementptr8BI
G
	full_text:
8
6%43 = getelementptr inbounds float, float* %5, i64 %12
%i648B

	full_text
	
i64 %12
Mstore8BB
@
	full_text3
1
/store float %42, float* %43, align 4, !tbaa !12
)float8B

	full_text

	float %42
+float*8B

	full_text


float* %43
'br8B

	full_text

br label %72
Bphi8B9
7
	full_text*
(
&%45 = phi i64 [ 0, %23 ], [ %69, %44 ]
%i648B

	full_text
	
i64 %69
Ophi8BF
D
	full_text7
5
3%46 = phi float [ 0.000000e+00, %23 ], [ %68, %44 ]
)float8B

	full_text

	float %68
Dphi8B;
9
	full_text,
*
(%47 = phi i64 [ %24, %23 ], [ %70, %44 ]
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %70
6mul8B-
+
	full_text

%48 = mul nsw i64 %45, %17
%i648B

	full_text
	
i64 %45
%i648B

	full_text
	
i64 %17
6add8B-
+
	full_text

%49 = add nsw i64 %48, %19
%i648B

	full_text
	
i64 %48
%i648B

	full_text
	
i64 %19
\getelementptr8BI
G
	full_text:
8
6%50 = getelementptr inbounds float, float* %0, i64 %49
%i648B

	full_text
	
i64 %49
Mload8BC
A
	full_text4
2
0%51 = load float, float* %50, align 4, !tbaa !12
+float*8B

	full_text


float* %50
Xgetelementptr8BE
C
	full_text6
4
2%52 = getelementptr inbounds i32, i32* %2, i64 %49
%i648B

	full_text
	
i64 %49
Hload8B>
<
	full_text/
-
+%53 = load i32, i32* %52, align 4, !tbaa !8
'i32*8B

	full_text


i32* %52
6sext8B,
*
	full_text

%54 = sext i32 %53 to i64
%i328B

	full_text
	
i32 %53
\getelementptr8BI
G
	full_text:
8
6%55 = getelementptr inbounds float, float* %1, i64 %54
%i648B

	full_text
	
i64 %54
Mload8BC
A
	full_text4
2
0%56 = load float, float* %55, align 4, !tbaa !12
+float*8B

	full_text


float* %55
ecall8B[
Y
	full_textL
J
H%57 = tail call float @llvm.fmuladd.f32(float %51, float %56, float %46)
)float8B

	full_text

	float %51
)float8B

	full_text

	float %56
)float8B

	full_text

	float %46
.or8B&
$
	full_text

%58 = or i64 %45, 1
%i648B

	full_text
	
i64 %45
6mul8B-
+
	full_text

%59 = mul nsw i64 %58, %17
%i648B

	full_text
	
i64 %58
%i648B

	full_text
	
i64 %17
6add8B-
+
	full_text

%60 = add nsw i64 %59, %19
%i648B

	full_text
	
i64 %59
%i648B

	full_text
	
i64 %19
\getelementptr8BI
G
	full_text:
8
6%61 = getelementptr inbounds float, float* %0, i64 %60
%i648B

	full_text
	
i64 %60
Mload8BC
A
	full_text4
2
0%62 = load float, float* %61, align 4, !tbaa !12
+float*8B

	full_text


float* %61
Xgetelementptr8BE
C
	full_text6
4
2%63 = getelementptr inbounds i32, i32* %2, i64 %60
%i648B

	full_text
	
i64 %60
Hload8B>
<
	full_text/
-
+%64 = load i32, i32* %63, align 4, !tbaa !8
'i32*8B

	full_text


i32* %63
6sext8B,
*
	full_text

%65 = sext i32 %64 to i64
%i328B

	full_text
	
i32 %64
\getelementptr8BI
G
	full_text:
8
6%66 = getelementptr inbounds float, float* %1, i64 %65
%i648B

	full_text
	
i64 %65
Mload8BC
A
	full_text4
2
0%67 = load float, float* %66, align 4, !tbaa !12
+float*8B

	full_text


float* %66
ecall8B[
Y
	full_textL
J
H%68 = tail call float @llvm.fmuladd.f32(float %62, float %67, float %57)
)float8B

	full_text

	float %62
)float8B

	full_text

	float %67
)float8B

	full_text

	float %57
4add8B+
)
	full_text

%69 = add nsw i64 %45, 2
%i648B

	full_text
	
i64 %45
1add8B(
&
	full_text

%70 = add i64 %47, -2
%i648B

	full_text
	
i64 %47
5icmp8B+
)
	full_text

%71 = icmp eq i64 %70, 0
%i648B

	full_text
	
i64 %70
:br8B2
0
	full_text#
!
br i1 %71, label %25, label %44
#i18B

	full_text


i1 %71
$ret8B

	full_text


ret void
*float*8	B

	full_text

	float* %0
&i32*8	B

	full_text
	
i32* %3
&i32*8	B

	full_text
	
i32* %2
*float*8	B

	full_text

	float* %5
$i328	B

	full_text


i32 %4
*float*8	B

	full_text

	float* %1
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
$i648	B

	full_text


i64 32
#i648	B

	full_text	

i64 2
#i328	B

	full_text	

i32 1
$i648	B

	full_text


i64 -2
+float8	B

	full_text

float undef
2float8	B%
#
	full_text

float 0.000000e+00
#i648	B

	full_text	

i64 1
#i648	B

	full_text	

i64 0
#i328	B

	full_text	

i32 0      	  
 

                     " !# !! $& %% '( '' )* )) +, ++ -. -0 /1 // 23 24 22 56 55 78 77 9: 99 ;< ;; => == ?@ ?? AB AA CD CE CF CC GI HJ HH KL KK MN MO MM PR QQ ST SS UV UW UU XY XZ XX [\ [] [[ ^_ ^^ `a `` bc bb de dd fg ff hi hh jk jj lm ln lo ll pq pp rs rt rr uv uw uu xy xx z{ zz |} || ~ ~~ ÄÅ ÄÄ Ç
É ÇÇ ÑÖ ÑÑ Üá Ü
à Ü
â ÜÜ äã ää åç åå éè éé êë êì 5ì ^ì xî ï 9ï bï |ñ K	ó ó ò ?ò hò Ç    	 
            " #Ü &ä (Ü * ,+ .' 0 1/ 3 42 65 82 :9 <; >= @? B7 DA E) F% IC J
 LH NK Oä RÜ T! Vå WQ Y ZX \ ][ _^ a[ cb ed gf ih k` mj nS oQ qp s tr v wu yx {u }| ~ ÅÄ ÉÇ Öz áÑ àl âQ ãU çå èé ë  í  H % !P í- H- /$ QG Hê %ê Q í öö ôô ôô l öö lÜ öö ÜC öö C	õ 	õ 
	õ 	õ 
ú ä	ù 
û åü %† )† H† S	° 	° p¢ '	¢ +¢ Q
¢ é£ 	£ "
spmv_ellpackr_kernel"
_Z13get_global_idj"
llvm.fmuladd.f32*†
'shoc-1.1.5-Spmv-spmv_ellpackr_kernel.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02Ä

wgsize
Ä
 
transfer_bytes_log1p
¿$hA

devmap_label
 

transfer_bytes
Ùçz

wgsize_log1p
¿$hA