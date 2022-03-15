

[external]
JcallBB
@
	full_text3
1
/%5 = tail call i64 @_Z12get_local_idj(i32 0) #3
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
0shlB)
'
	full_text

%9 = shl nsw i32 %8, 1
"i32B

	full_text


i32 %8
3sextB+
)
	full_text

%10 = sext i32 %9 to i64
"i32B

	full_text


i32 %9
ZgetelementptrBI
G
	full_text:
8
6%11 = getelementptr inbounds float, float* %1, i64 %10
#i64B

	full_text
	
i64 %10
>bitcastB3
1
	full_text$
"
 %12 = bitcast float* %11 to i32*
)float*B

	full_text


float* %11
FloadB>
<
	full_text/
-
+%13 = load i32, i32* %12, align 4, !tbaa !8
%i32*B

	full_text


i32* %12
1shlB*
(
	full_text

%14 = shl nsw i32 %6, 1
"i32B

	full_text


i32 %6
4sextB,
*
	full_text

%15 = sext i32 %14 to i64
#i32B

	full_text
	
i32 %14
ZgetelementptrBI
G
	full_text:
8
6%16 = getelementptr inbounds float, float* %2, i64 %15
#i64B

	full_text
	
i64 %15
>bitcastB3
1
	full_text$
"
 %17 = bitcast float* %16 to i32*
)float*B

	full_text


float* %16
FstoreB=
;
	full_text.
,
*store i32 %13, i32* %17, align 4, !tbaa !8
#i32B

	full_text
	
i32 %13
%i32*B

	full_text


i32* %17
+orB%
#
	full_text

%18 = or i32 %9, 1
"i32B

	full_text


i32 %9
4sextB,
*
	full_text

%19 = sext i32 %18 to i64
#i32B

	full_text
	
i32 %18
ZgetelementptrBI
G
	full_text:
8
6%20 = getelementptr inbounds float, float* %1, i64 %19
#i64B

	full_text
	
i64 %19
>bitcastB3
1
	full_text$
"
 %21 = bitcast float* %20 to i32*
)float*B

	full_text


float* %20
FloadB>
<
	full_text/
-
+%22 = load i32, i32* %21, align 4, !tbaa !8
%i32*B

	full_text


i32* %21
,orB&
$
	full_text

%23 = or i32 %14, 1
#i32B

	full_text
	
i32 %14
4sextB,
*
	full_text

%24 = sext i32 %23 to i64
#i32B

	full_text
	
i32 %23
ZgetelementptrBI
G
	full_text:
8
6%25 = getelementptr inbounds float, float* %2, i64 %24
#i64B

	full_text
	
i64 %24
>bitcastB3
1
	full_text$
"
 %26 = bitcast float* %25 to i32*
)float*B

	full_text


float* %25
FstoreB=
;
	full_text.
,
*store i32 %22, i32* %26, align 4, !tbaa !8
#i32B

	full_text
	
i32 %22
%i32*B

	full_text


i32* %26
@callB8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #4
IloadBA
?
	full_text2
0
.%27 = load float, float* %2, align 4, !tbaa !8
3icmpB+
)
	full_text

%28 = icmp ugt i32 %3, 1
8brB2
0
	full_text#
!
br i1 %28, label %29, label %33
!i1B

	full_text


i1 %28
Zgetelementptr8BG
E
	full_text8
6
4%30 = getelementptr inbounds float, float* %2, i64 1
Lload8BB
@
	full_text3
1
/%31 = load float, float* %30, align 4, !tbaa !8
+float*8B

	full_text


float* %30
6fadd8B,
*
	full_text

%32 = fadd float %27, %31
)float8B

	full_text

	float %27
)float8B

	full_text

	float %31
'br8B

	full_text

br label %35
4icmp8B*
(
	full_text

%34 = icmp eq i32 %6, 0
$i328B

	full_text


i32 %6
:br8B2
0
	full_text#
!
br i1 %34, label %58, label %60
#i18B

	full_text


i1 %34
Bphi8B9
7
	full_text*
(
&%36 = phi i32 [ 1, %29 ], [ %56, %53 ]
%i328B

	full_text
	
i32 %56
Fphi8B=
;
	full_text.
,
*%37 = phi float [ %32, %29 ], [ %55, %53 ]
)float8B

	full_text

	float %32
)float8B

	full_text

	float %55
Fphi8B=
;
	full_text.
,
*%38 = phi float [ %27, %29 ], [ %54, %53 ]
)float8B

	full_text

	float %27
)float8B

	full_text

	float %54
8icmp8B.
,
	full_text

%39 = icmp slt i32 %14, %36
%i328B

	full_text
	
i32 %14
%i328B

	full_text
	
i32 %36
:br8B2
0
	full_text#
!
br i1 %39, label %53, label %40
#i18B

	full_text


i1 %39
6sub8B-
+
	full_text

%41 = sub nsw i32 %14, %36
%i328B

	full_text
	
i32 %14
%i328B

	full_text
	
i32 %36
6sext8B,
*
	full_text

%42 = sext i32 %41 to i64
%i328B

	full_text
	
i32 %41
\getelementptr8BI
G
	full_text:
8
6%43 = getelementptr inbounds float, float* %2, i64 %42
%i648B

	full_text
	
i64 %42
Lload8BB
@
	full_text3
1
/%44 = load float, float* %43, align 4, !tbaa !8
+float*8B

	full_text


float* %43
Lload8BB
@
	full_text3
1
/%45 = load float, float* %16, align 4, !tbaa !8
+float*8B

	full_text


float* %16
6fadd8B,
*
	full_text

%46 = fadd float %44, %45
)float8B

	full_text

	float %44
)float8B

	full_text

	float %45
6sub8B-
+
	full_text

%47 = sub nsw i32 %23, %36
%i328B

	full_text
	
i32 %23
%i328B

	full_text
	
i32 %36
6sext8B,
*
	full_text

%48 = sext i32 %47 to i64
%i328B

	full_text
	
i32 %47
\getelementptr8BI
G
	full_text:
8
6%49 = getelementptr inbounds float, float* %2, i64 %48
%i648B

	full_text
	
i64 %48
Lload8BB
@
	full_text3
1
/%50 = load float, float* %49, align 4, !tbaa !8
+float*8B

	full_text


float* %49
Lload8BB
@
	full_text3
1
/%51 = load float, float* %25, align 4, !tbaa !8
+float*8B

	full_text


float* %25
6fadd8B,
*
	full_text

%52 = fadd float %50, %51
)float8B

	full_text

	float %50
)float8B

	full_text

	float %51
'br8B

	full_text

br label %53
Fphi8B=
;
	full_text.
,
*%54 = phi float [ %46, %40 ], [ %38, %35 ]
)float8B

	full_text

	float %46
)float8B

	full_text

	float %38
Fphi8B=
;
	full_text.
,
*%55 = phi float [ %52, %40 ], [ %37, %35 ]
)float8B

	full_text

	float %52
)float8B

	full_text

	float %37
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #4
Lstore8BA
?
	full_text2
0
.store float %54, float* %16, align 4, !tbaa !8
)float8B

	full_text

	float %54
+float*8B

	full_text


float* %16
Lstore8BA
?
	full_text2
0
.store float %55, float* %25, align 4, !tbaa !8
)float8B

	full_text

	float %55
+float*8B

	full_text


float* %25
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #4
4shl8B+
)
	full_text

%56 = shl nsw i32 %36, 1
%i328B

	full_text
	
i32 %36
7icmp8B-
+
	full_text

%57 = icmp ult i32 %56, %3
%i328B

	full_text
	
i32 %56
:br8B2
0
	full_text#
!
br i1 %57, label %35, label %33
#i18B

	full_text


i1 %57
\getelementptr8BI
G
	full_text:
8
6%59 = getelementptr inbounds float, float* %0, i64 %10
%i648B

	full_text
	
i64 %10
Ustore8BJ
H
	full_text;
9
7store float 0.000000e+00, float* %59, align 4, !tbaa !8
+float*8B

	full_text


float* %59
'br8B

	full_text

br label %68
5add8B,
*
	full_text

%61 = add nsw i32 %14, -1
%i328B

	full_text
	
i32 %14
6sext8B,
*
	full_text

%62 = sext i32 %61 to i64
%i328B

	full_text
	
i32 %61
\getelementptr8BI
G
	full_text:
8
6%63 = getelementptr inbounds float, float* %2, i64 %62
%i648B

	full_text
	
i64 %62
@bitcast8B3
1
	full_text$
"
 %64 = bitcast float* %63 to i32*
+float*8B

	full_text


float* %63
Hload8B>
<
	full_text/
-
+%65 = load i32, i32* %64, align 4, !tbaa !8
'i32*8B

	full_text


i32* %64
\getelementptr8BI
G
	full_text:
8
6%66 = getelementptr inbounds float, float* %0, i64 %10
%i648B

	full_text
	
i64 %10
@bitcast8B3
1
	full_text$
"
 %67 = bitcast float* %66 to i32*
+float*8B

	full_text


float* %66
Hstore8B=
;
	full_text.
,
*store i32 %65, i32* %67, align 4, !tbaa !8
%i328B

	full_text
	
i32 %65
'i32*8B

	full_text


i32* %67
'br8B

	full_text

br label %68
Hload8B>
<
	full_text/
-
+%69 = load i32, i32* %17, align 4, !tbaa !8
'i32*8B

	full_text


i32* %17
\getelementptr8BI
G
	full_text:
8
6%70 = getelementptr inbounds float, float* %0, i64 %19
%i648B

	full_text
	
i64 %19
@bitcast8B3
1
	full_text$
"
 %71 = bitcast float* %70 to i32*
+float*8B

	full_text


float* %70
Hstore8B=
;
	full_text.
,
*store i32 %69, i32* %71, align 4, !tbaa !8
%i328B

	full_text
	
i32 %69
'i32*8B

	full_text


i32* %71
$ret8B

	full_text


ret void
$i328	B

	full_text


i32 %3
*float*8	B

	full_text

	float* %1
*float*8	B

	full_text

	float* %0
*float*8	B

	full_text

	float* %2
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
-; undefined function B

	full_text

 
#i648	B

	full_text	

i64 1
#i328	B

	full_text	

i32 1
2float8	B%
#
	full_text

float 0.000000e+00
#i328	B

	full_text	

i32 0
$i328	B

	full_text


i32 -1        	
 		                       !    "# "" $% $$ &' && () (( *+ ** ,- ,, ./ .0 .. 11 22 33 45 46 78 77 9: 9; 99 <> == ?@ ?B AA CD CE CC FG FH FF IJ IK II LM LO NP NN QR QQ ST SS UV UU WX WW YZ Y[ YY \] \^ \\ _` __ ab aa cd cc ef ee gh gi gg jl km kk no np nn qq rs rt rr uv uw uu xx yz yy {| {{ }~ }	Ä  Å
Ç ÅÅ ÉÖ ÑÑ Üá ÜÜ à
â àà äã ää åç åå é
è éé êë êê íì í
î íí ïó ññ ò
ô òò öõ öö úù ú
û úú ü† 3	† {° °  ¢ ¢ é¢ ò£ £ *£ 2£ 6£ S£ a£ à    
	            !  #" % '& )( +* -$ /, 03 56 82 :7 ; >= @y B9 Dn E2 Gk H JA KI M OA PN RQ TS V XU ZW [& ]A ^\ `_ ba d* fc he iY lF mg oC pk s tn v* wA zy |{ ~	 Ä Ç ÖÑ áÜ âà ãä ç	 èé ëå ìê î ó ôò õñ ùö û4 64 =< A? ? ÑL kL NÉ ñï ñ} A} =j k •• §§ ¶¶ ü •• 1 ¶¶ 1 §§ x ¶¶ xq ¶¶ q	ß 6	® 	® 	® 	® &® 1	® 3® A® q® x	® y© Å™ ™ 	™ =
´ Ñ"
	prefixSum"
_Z12get_local_idj"
_Z13get_global_idj"
_Z7barrierj*ï
ScanLargeArrays-prefixSum.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02
 
transfer_bytes_log1p
W∞GA

devmap_label
 

transfer_bytes
Äà

wgsize_log1p
W∞GA

wgsize
@