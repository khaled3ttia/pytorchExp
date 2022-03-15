

[external]
JcallBB
@
	full_text3
1
/%6 = tail call i64 @_Z12get_group_idj(i32 0) #4
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
JcallBB
@
	full_text3
1
/%8 = tail call i64 @_Z12get_group_idj(i32 1) #4
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
KcallBC
A
	full_text4
2
0%10 = tail call i64 @_Z12get_local_idj(i32 0) #4
6truncB-
+
	full_text

%11 = trunc i64 %10 to i32
#i64B

	full_text
	
i64 %10
KcallBC
A
	full_text4
2
0%12 = tail call i64 @_Z12get_local_idj(i32 1) #4
6truncB-
+
	full_text

%13 = trunc i64 %12 to i32
#i64B

	full_text
	
i64 %12
-shlB&
$
	full_text

%14 = shl i32 %9, 6
"i32B

	full_text


i32 %9
-shlB&
$
	full_text

%15 = shl i32 %7, 6
"i32B

	full_text


i32 %7
.addB'
%
	full_text

%16 = add i32 %4, 64
0addB)
'
	full_text

%17 = add i32 %16, %15
#i32B

	full_text
	
i32 %16
#i32B

	full_text
	
i32 %15
3addB,
*
	full_text

%18 = add nsw i32 %13, %4
#i32B

	full_text
	
i32 %13
3mulB,
*
	full_text

%19 = mul nsw i32 %18, %3
#i32B

	full_text
	
i32 %18
0addB)
'
	full_text

%20 = add i32 %17, %11
#i32B

	full_text
	
i32 %17
#i32B

	full_text
	
i32 %11
0addB)
'
	full_text

%21 = add i32 %20, %19
#i32B

	full_text
	
i32 %20
#i32B

	full_text
	
i32 %19
4sextB,
*
	full_text

%22 = sext i32 %21 to i64
#i32B

	full_text
	
i32 %21
ZgetelementptrBI
G
	full_text:
8
6%23 = getelementptr inbounds float, float* %0, i64 %22
#i64B

	full_text
	
i64 %22
>bitcastB3
1
	full_text$
"
 %24 = bitcast float* %23 to i32*
)float*B

	full_text


float* %23
FloadB>
<
	full_text/
-
+%25 = load i32, i32* %24, align 4, !tbaa !8
%i32*B

	full_text


i32* %24
2shlB+
)
	full_text

%26 = shl nsw i32 %13, 6
#i32B

	full_text
	
i32 %13
4addB-
+
	full_text

%27 = add nsw i32 %26, %11
#i32B

	full_text
	
i32 %26
#i32B

	full_text
	
i32 %11
4sextB,
*
	full_text

%28 = sext i32 %27 to i64
#i32B

	full_text
	
i32 %27
ZgetelementptrBI
G
	full_text:
8
6%29 = getelementptr inbounds float, float* %1, i64 %28
#i64B

	full_text
	
i64 %28
>bitcastB3
1
	full_text$
"
 %30 = bitcast float* %29 to i32*
)float*B

	full_text


float* %29
FstoreB=
;
	full_text.
,
*store i32 %25, i32* %30, align 4, !tbaa !8
#i32B

	full_text
	
i32 %25
%i32*B

	full_text


i32* %30
0addB)
'
	full_text

%31 = add i32 %16, %14
#i32B

	full_text
	
i32 %16
#i32B

	full_text
	
i32 %14
0addB)
'
	full_text

%32 = add i32 %31, %13
#i32B

	full_text
	
i32 %31
#i32B

	full_text
	
i32 %13
3mulB,
*
	full_text

%33 = mul nsw i32 %32, %3
#i32B

	full_text
	
i32 %32
/addB(
&
	full_text

%34 = add i32 %11, %4
#i32B

	full_text
	
i32 %11
0addB)
'
	full_text

%35 = add i32 %34, %33
#i32B

	full_text
	
i32 %34
#i32B

	full_text
	
i32 %33
4sextB,
*
	full_text

%36 = sext i32 %35 to i64
#i32B

	full_text
	
i32 %35
ZgetelementptrBI
G
	full_text:
8
6%37 = getelementptr inbounds float, float* %0, i64 %36
#i64B

	full_text
	
i64 %36
>bitcastB3
1
	full_text$
"
 %38 = bitcast float* %37 to i32*
)float*B

	full_text


float* %37
FloadB>
<
	full_text/
-
+%39 = load i32, i32* %38, align 4, !tbaa !8
%i32*B

	full_text


i32* %38
ZgetelementptrBI
G
	full_text:
8
6%40 = getelementptr inbounds float, float* %2, i64 %28
#i64B

	full_text
	
i64 %28
>bitcastB3
1
	full_text$
"
 %41 = bitcast float* %40 to i32*
)float*B

	full_text


float* %40
FstoreB=
;
	full_text.
,
*store i32 %39, i32* %41, align 4, !tbaa !8
#i32B

	full_text
	
i32 %39
%i32*B

	full_text


i32* %41
@callB8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
4sextB,
*
	full_text

%42 = sext i32 %26 to i64
#i32B

	full_text
	
i32 %26
/shlB(
&
	full_text

%43 = shl i64 %10, 32
#i64B

	full_text
	
i64 %10
7ashrB/
-
	full_text 

%44 = ashr exact i64 %43, 32
#i64B

	full_text
	
i64 %43
%brB

	full_text

br label %45
Aphi8B8
6
	full_text)
'
%%46 = phi i64 [ 0, %5 ], [ %65, %45 ]
%i648B

	full_text
	
i64 %65
Nphi8BE
C
	full_text6
4
2%47 = phi float [ 0.000000e+00, %5 ], [ %64, %45 ]
)float8B

	full_text

	float %64
:add8B1
/
	full_text"
 
%48 = add nuw nsw i64 %46, %42
%i648B

	full_text
	
i64 %46
%i648B

	full_text
	
i64 %42
\getelementptr8BI
G
	full_text:
8
6%49 = getelementptr inbounds float, float* %2, i64 %48
%i648B

	full_text
	
i64 %48
Lload8BB
@
	full_text3
1
/%50 = load float, float* %49, align 4, !tbaa !8
+float*8B

	full_text


float* %49
0shl8B'
%
	full_text

%51 = shl i64 %46, 6
%i648B

	full_text
	
i64 %46
6add8B-
+
	full_text

%52 = add nsw i64 %51, %44
%i648B

	full_text
	
i64 %51
%i648B

	full_text
	
i64 %44
\getelementptr8BI
G
	full_text:
8
6%53 = getelementptr inbounds float, float* %1, i64 %52
%i648B

	full_text
	
i64 %52
Lload8BB
@
	full_text3
1
/%54 = load float, float* %53, align 4, !tbaa !8
+float*8B

	full_text


float* %53
ecall8B[
Y
	full_textL
J
H%55 = tail call float @llvm.fmuladd.f32(float %50, float %54, float %47)
)float8B

	full_text

	float %50
)float8B

	full_text

	float %54
)float8B

	full_text

	float %47
.or8B&
$
	full_text

%56 = or i64 %46, 1
%i648B

	full_text
	
i64 %46
:add8B1
/
	full_text"
 
%57 = add nuw nsw i64 %56, %42
%i648B

	full_text
	
i64 %56
%i648B

	full_text
	
i64 %42
\getelementptr8BI
G
	full_text:
8
6%58 = getelementptr inbounds float, float* %2, i64 %57
%i648B

	full_text
	
i64 %57
Lload8BB
@
	full_text3
1
/%59 = load float, float* %58, align 4, !tbaa !8
+float*8B

	full_text


float* %58
0shl8B'
%
	full_text

%60 = shl i64 %56, 6
%i648B

	full_text
	
i64 %56
6add8B-
+
	full_text

%61 = add nsw i64 %60, %44
%i648B

	full_text
	
i64 %60
%i648B

	full_text
	
i64 %44
\getelementptr8BI
G
	full_text:
8
6%62 = getelementptr inbounds float, float* %1, i64 %61
%i648B

	full_text
	
i64 %61
Lload8BB
@
	full_text3
1
/%63 = load float, float* %62, align 4, !tbaa !8
+float*8B

	full_text


float* %62
ecall8B[
Y
	full_textL
J
H%64 = tail call float @llvm.fmuladd.f32(float %59, float %63, float %55)
)float8B

	full_text

	float %59
)float8B

	full_text

	float %63
)float8B

	full_text

	float %55
4add8B+
)
	full_text

%65 = add nsw i64 %46, 2
%i648B

	full_text
	
i64 %46
6icmp8B,
*
	full_text

%66 = icmp eq i64 %65, 64
%i648B

	full_text
	
i64 %65
:br8B2
0
	full_text#
!
br i1 %66, label %67, label %45
#i18B

	full_text


i1 %66
2add8B)
'
	full_text

%68 = add i32 %20, %33
%i328B

	full_text
	
i32 %20
%i328B

	full_text
	
i32 %33
6sext8B,
*
	full_text

%69 = sext i32 %68 to i64
%i328B

	full_text
	
i32 %68
\getelementptr8BI
G
	full_text:
8
6%70 = getelementptr inbounds float, float* %0, i64 %69
%i648B

	full_text
	
i64 %69
Lload8BB
@
	full_text3
1
/%71 = load float, float* %70, align 4, !tbaa !8
+float*8B

	full_text


float* %70
6fsub8B,
*
	full_text

%72 = fsub float %71, %64
)float8B

	full_text

	float %71
)float8B

	full_text

	float %64
Lstore8BA
?
	full_text2
0
.store float %72, float* %70, align 4, !tbaa !8
)float8B

	full_text

	float %72
+float*8B

	full_text


float* %70
$ret8B

	full_text


ret void
*float*8B

	full_text

	float* %2
$i328B

	full_text


i32 %4
*float*8B

	full_text

	float* %0
*float*8B

	full_text

	float* %1
$i328B

	full_text


i32 %3
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
-; undefined function B

	full_text

 
2float8B%
#
	full_text

float 0.000000e+00
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 6
#i648B

	full_text	

i64 1
#i328B

	full_text	

i32 1
$i648B

	full_text


i64 32
#i648B

	full_text	

i64 2
$i648B

	full_text


i64 64
$i328B

	full_text


i32 64
#i648B

	full_text	

i64 0
#i328B

	full_text	

i32 6       	  

                        !" !! #$ ## %& %% '( '' )* )+ )) ,- ,, ./ .. 01 00 23 24 22 56 57 55 89 8: 88 ;< ;; => == ?@ ?A ?? BC BB DE DD FG FF HI HH JK JJ LM LL NO NP NN QQ RS RR TU TT VW VV XZ YY [\ [[ ]^ ]_ ]] `a `` bc bb de dd fg fh ff ij ii kl kk mn mo mp mm qr qq st su ss vw vv xy xx z{ zz |} |~ || 	Ä  ÅÇ ÅÅ ÉÑ É
Ö É
Ü ÉÉ áà áá âä ââ ãå ãé ç
è çç êë êê í
ì íí îï îî ñó ñ
ò ññ ôö ô
õ ôô úù Jù `ù vû 	û 	û =ü !ü Dü í† .† i† 	° 	° ;   	
              "! $# & (' * +) -, /. 1% 30 4 6 75 9 :8 < >= @; A? CB ED GF I, KJ MH OL P' S UT Wá ZÉ \Y ^R _] a` cY ed gV hf ji lb nk o[ pY rq tR us wv yq {z }V ~| Ä Çx ÑÅ Öm ÜY àá äâ å é; èç ëê ìí ïî óÉ òñ öí õX Yã çã Y ¢¢ ££ ú §§ •• ¢¢  ££  ¢¢ 
 ££ 
Q §§ Qm •• mÉ •• É¶ [ß ß 	® d	® z	© q™ ™ 
™ Q	´ T	´ V
¨ á
≠ â	Æ Ø Y	∞ 	∞ 	∞ '"
lud_internal"
_Z12get_group_idj"
_Z12get_local_idj"
_Z7barrierj"
llvm.fmuladd.f32*ò
rodinia-3.1-lud-lud_internal.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02Å

transfer_bytes
ÄÄÄ

wgsize
Ä

devmap_label

 
transfer_bytes_log1p
·¸sA

wgsize_log1p
·¸sA