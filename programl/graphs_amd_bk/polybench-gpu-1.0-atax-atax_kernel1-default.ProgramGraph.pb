

[external]
KcallBC
A
	full_text4
2
0%6 = tail call i64 @_Z13get_global_idj(i32 0) #3
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
3icmpB+
)
	full_text

%8 = icmp slt i32 %7, %3
"i32B

	full_text


i32 %7
2icmpB*
(
	full_text

%9 = icmp sgt i32 %4, 0
-andB&
$
	full_text

%10 = and i1 %8, %9
 i1B

	full_text	

i1 %8
 i1B

	full_text	

i1 %9
8brB2
0
	full_text#
!
br i1 %10, label %11, label %76
!i1B

	full_text


i1 %10
4mul8B+
)
	full_text

%12 = mul nsw i32 %7, %4
$i328B

	full_text


i32 %7
0shl8B'
%
	full_text

%13 = shl i64 %6, 32
$i648B

	full_text


i64 %6
9ashr8B/
-
	full_text 

%14 = ashr exact i64 %13, 32
%i648B

	full_text
	
i64 %13
\getelementptr8BI
G
	full_text:
8
6%15 = getelementptr inbounds float, float* %2, i64 %14
%i648B

	full_text
	
i64 %14
6sext8B,
*
	full_text

%16 = sext i32 %12 to i64
%i328B

	full_text
	
i32 %12
Lload8BB
@
	full_text3
1
/%17 = load float, float* %15, align 4, !tbaa !9
+float*8B

	full_text


float* %15
5zext8B+
)
	full_text

%18 = zext i32 %4 to i64
5add8B,
*
	full_text

%19 = add nsw i64 %18, -1
%i648B

	full_text
	
i64 %18
0and8B'
%
	full_text

%20 = and i64 %18, 3
%i648B

	full_text
	
i64 %18
6icmp8B,
*
	full_text

%21 = icmp ult i64 %19, 3
%i648B

	full_text
	
i64 %19
:br8B2
0
	full_text#
!
br i1 %21, label %58, label %22
#i18B

	full_text


i1 %21
6sub8B-
+
	full_text

%23 = sub nsw i64 %18, %20
%i648B

	full_text
	
i64 %18
%i648B

	full_text
	
i64 %20
'br8B

	full_text

br label %24
Fphi8B=
;
	full_text.
,
*%25 = phi float [ %17, %22 ], [ %54, %24 ]
)float8B

	full_text

	float %17
)float8B

	full_text

	float %54
Bphi8B9
7
	full_text*
(
&%26 = phi i64 [ 0, %22 ], [ %55, %24 ]
%i648B

	full_text
	
i64 %55
Dphi8B;
9
	full_text,
*
(%27 = phi i64 [ %23, %22 ], [ %56, %24 ]
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %56
6add8B-
+
	full_text

%28 = add nsw i64 %26, %16
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %16
\getelementptr8BI
G
	full_text:
8
6%29 = getelementptr inbounds float, float* %0, i64 %28
%i648B

	full_text
	
i64 %28
Lload8BB
@
	full_text3
1
/%30 = load float, float* %29, align 4, !tbaa !9
+float*8B

	full_text


float* %29
\getelementptr8BI
G
	full_text:
8
6%31 = getelementptr inbounds float, float* %1, i64 %26
%i648B

	full_text
	
i64 %26
Lload8BB
@
	full_text3
1
/%32 = load float, float* %31, align 4, !tbaa !9
+float*8B

	full_text


float* %31
ecall8B[
Y
	full_textL
J
H%33 = tail call float @llvm.fmuladd.f32(float %30, float %32, float %25)
)float8B

	full_text

	float %30
)float8B

	full_text

	float %32
)float8B

	full_text

	float %25
Lstore8BA
?
	full_text2
0
.store float %33, float* %15, align 4, !tbaa !9
)float8B

	full_text

	float %33
+float*8B

	full_text


float* %15
.or8B&
$
	full_text

%34 = or i64 %26, 1
%i648B

	full_text
	
i64 %26
6add8B-
+
	full_text

%35 = add nsw i64 %34, %16
%i648B

	full_text
	
i64 %34
%i648B

	full_text
	
i64 %16
\getelementptr8BI
G
	full_text:
8
6%36 = getelementptr inbounds float, float* %0, i64 %35
%i648B

	full_text
	
i64 %35
Lload8BB
@
	full_text3
1
/%37 = load float, float* %36, align 4, !tbaa !9
+float*8B

	full_text


float* %36
\getelementptr8BI
G
	full_text:
8
6%38 = getelementptr inbounds float, float* %1, i64 %34
%i648B

	full_text
	
i64 %34
Lload8BB
@
	full_text3
1
/%39 = load float, float* %38, align 4, !tbaa !9
+float*8B

	full_text


float* %38
ecall8B[
Y
	full_textL
J
H%40 = tail call float @llvm.fmuladd.f32(float %37, float %39, float %33)
)float8B

	full_text

	float %37
)float8B

	full_text

	float %39
)float8B

	full_text

	float %33
Lstore8BA
?
	full_text2
0
.store float %40, float* %15, align 4, !tbaa !9
)float8B

	full_text

	float %40
+float*8B

	full_text


float* %15
.or8B&
$
	full_text

%41 = or i64 %26, 2
%i648B

	full_text
	
i64 %26
6add8B-
+
	full_text

%42 = add nsw i64 %41, %16
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %16
\getelementptr8BI
G
	full_text:
8
6%43 = getelementptr inbounds float, float* %0, i64 %42
%i648B

	full_text
	
i64 %42
Lload8BB
@
	full_text3
1
/%44 = load float, float* %43, align 4, !tbaa !9
+float*8B

	full_text


float* %43
\getelementptr8BI
G
	full_text:
8
6%45 = getelementptr inbounds float, float* %1, i64 %41
%i648B

	full_text
	
i64 %41
Lload8BB
@
	full_text3
1
/%46 = load float, float* %45, align 4, !tbaa !9
+float*8B

	full_text


float* %45
ecall8B[
Y
	full_textL
J
H%47 = tail call float @llvm.fmuladd.f32(float %44, float %46, float %40)
)float8B

	full_text

	float %44
)float8B

	full_text

	float %46
)float8B

	full_text

	float %40
Lstore8BA
?
	full_text2
0
.store float %47, float* %15, align 4, !tbaa !9
)float8B

	full_text

	float %47
+float*8B

	full_text


float* %15
.or8B&
$
	full_text

%48 = or i64 %26, 3
%i648B

	full_text
	
i64 %26
6add8B-
+
	full_text

%49 = add nsw i64 %48, %16
%i648B

	full_text
	
i64 %48
%i648B

	full_text
	
i64 %16
\getelementptr8BI
G
	full_text:
8
6%50 = getelementptr inbounds float, float* %0, i64 %49
%i648B

	full_text
	
i64 %49
Lload8BB
@
	full_text3
1
/%51 = load float, float* %50, align 4, !tbaa !9
+float*8B

	full_text


float* %50
\getelementptr8BI
G
	full_text:
8
6%52 = getelementptr inbounds float, float* %1, i64 %48
%i648B

	full_text
	
i64 %48
Lload8BB
@
	full_text3
1
/%53 = load float, float* %52, align 4, !tbaa !9
+float*8B

	full_text


float* %52
ecall8B[
Y
	full_textL
J
H%54 = tail call float @llvm.fmuladd.f32(float %51, float %53, float %47)
)float8B

	full_text

	float %51
)float8B

	full_text

	float %53
)float8B

	full_text

	float %47
Lstore8BA
?
	full_text2
0
.store float %54, float* %15, align 4, !tbaa !9
)float8B

	full_text

	float %54
+float*8B

	full_text


float* %15
4add8B+
)
	full_text

%55 = add nsw i64 %26, 4
%i648B

	full_text
	
i64 %26
1add8B(
&
	full_text

%56 = add i64 %27, -4
%i648B

	full_text
	
i64 %27
5icmp8B+
)
	full_text

%57 = icmp eq i64 %56, 0
%i648B

	full_text
	
i64 %56
:br8B2
0
	full_text#
!
br i1 %57, label %58, label %24
#i18B

	full_text


i1 %57
Fphi8B=
;
	full_text.
,
*%59 = phi float [ %17, %11 ], [ %54, %24 ]
)float8B

	full_text

	float %17
)float8B

	full_text

	float %54
Bphi8B9
7
	full_text*
(
&%60 = phi i64 [ 0, %11 ], [ %55, %24 ]
%i648B

	full_text
	
i64 %55
5icmp8B+
)
	full_text

%61 = icmp eq i64 %20, 0
%i648B

	full_text
	
i64 %20
:br8B2
0
	full_text#
!
br i1 %61, label %76, label %62
#i18B

	full_text


i1 %61
'br8B

	full_text

br label %63
Fphi8B=
;
	full_text.
,
*%64 = phi float [ %59, %62 ], [ %72, %63 ]
)float8B

	full_text

	float %59
)float8B

	full_text

	float %72
Dphi8B;
9
	full_text,
*
(%65 = phi i64 [ %60, %62 ], [ %73, %63 ]
%i648B

	full_text
	
i64 %60
%i648B

	full_text
	
i64 %73
Dphi8B;
9
	full_text,
*
(%66 = phi i64 [ %20, %62 ], [ %74, %63 ]
%i648B

	full_text
	
i64 %20
%i648B

	full_text
	
i64 %74
6add8B-
+
	full_text

%67 = add nsw i64 %65, %16
%i648B

	full_text
	
i64 %65
%i648B

	full_text
	
i64 %16
\getelementptr8BI
G
	full_text:
8
6%68 = getelementptr inbounds float, float* %0, i64 %67
%i648B

	full_text
	
i64 %67
Lload8BB
@
	full_text3
1
/%69 = load float, float* %68, align 4, !tbaa !9
+float*8B

	full_text


float* %68
\getelementptr8BI
G
	full_text:
8
6%70 = getelementptr inbounds float, float* %1, i64 %65
%i648B

	full_text
	
i64 %65
Lload8BB
@
	full_text3
1
/%71 = load float, float* %70, align 4, !tbaa !9
+float*8B

	full_text


float* %70
ecall8B[
Y
	full_textL
J
H%72 = tail call float @llvm.fmuladd.f32(float %69, float %71, float %64)
)float8B

	full_text

	float %69
)float8B

	full_text

	float %71
)float8B

	full_text

	float %64
Lstore8BA
?
	full_text2
0
.store float %72, float* %15, align 4, !tbaa !9
)float8B

	full_text

	float %72
+float*8B

	full_text


float* %15
8add8B/
-
	full_text 

%73 = add nuw nsw i64 %65, 1
%i648B

	full_text
	
i64 %65
1add8B(
&
	full_text

%74 = add i64 %66, -1
%i648B

	full_text
	
i64 %66
5icmp8B+
)
	full_text

%75 = icmp eq i64 %74, 0
%i648B

	full_text
	
i64 %74
Jbr8BB
@
	full_text3
1
/br i1 %75, label %76, label %63, !llvm.loop !13
#i18B

	full_text


i1 %75
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %4
*float*8B

	full_text

	float* %0
$i328B

	full_text


i32 %3
*float*8B

	full_text

	float* %1
*float*8B
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
$i648B

	full_text


i64 32
#i648B

	full_text	

i64 1
#i648B

	full_text	

i64 4
#i648B

	full_text	

i64 2
$i648B

	full_text


i64 -1
#i648B

	full_text	

i64 3
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 0
$i648B

	full_text


i64 -4       	  
 
                     " !# !! $& %' %% () (( *+ *, ** -. -/ -- 01 00 23 22 45 44 67 66 89 8: 8; 88 <= <> << ?@ ?? AB AC AA DE DD FG FF HI HH JK JJ LM LN LO LL PQ PR PP ST SS UV UW UU XY XX Z[ ZZ \] \\ ^_ ^^ `a `b `c `` de df dd gh gg ij ik ii lm ll no nn pq pp rs rr tu tv tw tt xy xz xx {| {{ }~ }} Ä  ÅÇ ÅÑ É
Ö ÉÉ Ü
á ÜÜ àâ àà äã äé ç
è çç êë ê
í êê ìî ì
ï ìì ñó ñ
ò ññ ô
ö ôô õú õõ ù
û ùù ü† üü °¢ °
£ °
§ °° •¶ •
ß •• ®© ®® ™´ ™™ ¨≠ ¨¨ ÆØ Æ± 	± ± ≤ 0≤ D≤ X≤ l≤ ô	≥ ¥ 4¥ H¥ \¥ p¥ ùµ     	             " # &t '{ )! +} ,( . /- 10 3( 54 72 96 :% ;8 = >( @? B CA ED G? IH KF MJ N8 OL Q R( TS V WU YX [S ]\ _Z a^ bL c` e f( hg j ki ml og qp sn ur v` wt y z( |* ~} Ä Ç Ñt Ö{ á âà ãÉ é° èÜ ë® í î™ ïê ó òñ öô úê ûù †õ ¢ü £ç §° ¶ ßê ©ì ´™ ≠¨ Ø
 
 ∞ É !ä ∞ä å$ %å çÅ ÉÅ %Æ ∞Æ ç ∂∂ ∑∑ ∞` ∑∑ `t ∑∑ t ∂∂ ° ∑∑ °8 ∑∑ 8L ∑∑ L	∏ 	∏ 	π ?
π ®	∫ {	ª S	º 
º ™	Ω 	Ω 	Ω gæ 	æ ø (	ø ø Ü
ø à
ø ¨	¿ }"
atax_kernel1"
_Z13get_global_idj"
llvm.fmuladd.f32*ü
&polybench-gpu-1.0-atax-atax_kernel1.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Å

wgsize_log1p
3.êA
 
transfer_bytes_log1p
3.êA

wgsize
Ä

devmap_label


transfer_bytes
ÄÄÉ 