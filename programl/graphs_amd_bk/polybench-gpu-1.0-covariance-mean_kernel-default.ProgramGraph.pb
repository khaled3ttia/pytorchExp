

[external]
KcallBC
A
	full_text4
2
0%6 = tail call i64 @_Z13get_global_idj(i32 0) #2
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
6brB0
.
	full_text!

br i1 %8, label %9, label %75
 i1B

	full_text	

i1 %8
0shl8B'
%
	full_text

%10 = shl i64 %6, 32
$i648B

	full_text


i64 %6
9ashr8B/
-
	full_text 

%11 = ashr exact i64 %10, 32
%i648B

	full_text
	
i64 %10
\getelementptr8BI
G
	full_text:
8
6%12 = getelementptr inbounds float, float* %0, i64 %11
%i648B

	full_text
	
i64 %11
Ustore8BJ
H
	full_text;
9
7store float 0.000000e+00, float* %12, align 4, !tbaa !9
+float*8B

	full_text


float* %12
5icmp8B+
)
	full_text

%13 = icmp sgt i32 %4, 0
:br8B2
0
	full_text#
!
br i1 %13, label %14, label %72
#i18B

	full_text


i1 %13
5sext8B+
)
	full_text

%15 = sext i32 %3 to i64
0shl8B'
%
	full_text

%16 = shl i64 %6, 32
$i648B

	full_text


i64 %6
9ashr8B/
-
	full_text 

%17 = ashr exact i64 %16, 32
%i648B

	full_text
	
i64 %16
5zext8B+
)
	full_text

%18 = zext i32 %4 to i64
5add8B,
*
	full_text

%19 = add nsw i64 %18, -1
%i648B

	full_text
	
i64 %18
0and8B'
%
	full_text

%20 = and i64 %18, 3
%i648B

	full_text
	
i64 %18
6icmp8B,
*
	full_text

%21 = icmp ult i64 %19, 3
%i648B

	full_text
	
i64 %19
:br8B2
0
	full_text#
!
br i1 %21, label %54, label %22
#i18B

	full_text


i1 %21
6sub8B-
+
	full_text

%23 = sub nsw i64 %18, %20
%i648B

	full_text
	
i64 %18
%i648B

	full_text
	
i64 %20
'br8B

	full_text

br label %24
Ophi8BF
D
	full_text7
5
3%25 = phi float [ 0.000000e+00, %22 ], [ %50, %24 ]
)float8B

	full_text

	float %50
Bphi8B9
7
	full_text*
(
&%26 = phi i64 [ 0, %22 ], [ %51, %24 ]
%i648B

	full_text
	
i64 %51
Dphi8B;
9
	full_text,
*
(%27 = phi i64 [ %23, %22 ], [ %52, %24 ]
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %52
6mul8B-
+
	full_text

%28 = mul nsw i64 %26, %15
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %15
6add8B-
+
	full_text

%29 = add nsw i64 %28, %17
%i648B

	full_text
	
i64 %28
%i648B

	full_text
	
i64 %17
\getelementptr8BI
G
	full_text:
8
6%30 = getelementptr inbounds float, float* %1, i64 %29
%i648B

	full_text
	
i64 %29
Lload8BB
@
	full_text3
1
/%31 = load float, float* %30, align 4, !tbaa !9
+float*8B

	full_text


float* %30
6fadd8B,
*
	full_text

%32 = fadd float %31, %25
)float8B

	full_text

	float %31
)float8B

	full_text

	float %25
Lstore8BA
?
	full_text2
0
.store float %32, float* %12, align 4, !tbaa !9
)float8B

	full_text

	float %32
+float*8B

	full_text


float* %12
.or8B&
$
	full_text

%33 = or i64 %26, 1
%i648B

	full_text
	
i64 %26
6mul8B-
+
	full_text

%34 = mul nsw i64 %33, %15
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %15
6add8B-
+
	full_text

%35 = add nsw i64 %34, %17
%i648B

	full_text
	
i64 %34
%i648B

	full_text
	
i64 %17
\getelementptr8BI
G
	full_text:
8
6%36 = getelementptr inbounds float, float* %1, i64 %35
%i648B

	full_text
	
i64 %35
Lload8BB
@
	full_text3
1
/%37 = load float, float* %36, align 4, !tbaa !9
+float*8B

	full_text


float* %36
6fadd8B,
*
	full_text

%38 = fadd float %37, %32
)float8B

	full_text

	float %37
)float8B

	full_text

	float %32
Lstore8BA
?
	full_text2
0
.store float %38, float* %12, align 4, !tbaa !9
)float8B

	full_text

	float %38
+float*8B

	full_text


float* %12
.or8B&
$
	full_text

%39 = or i64 %26, 2
%i648B

	full_text
	
i64 %26
6mul8B-
+
	full_text

%40 = mul nsw i64 %39, %15
%i648B

	full_text
	
i64 %39
%i648B

	full_text
	
i64 %15
6add8B-
+
	full_text

%41 = add nsw i64 %40, %17
%i648B

	full_text
	
i64 %40
%i648B

	full_text
	
i64 %17
\getelementptr8BI
G
	full_text:
8
6%42 = getelementptr inbounds float, float* %1, i64 %41
%i648B

	full_text
	
i64 %41
Lload8BB
@
	full_text3
1
/%43 = load float, float* %42, align 4, !tbaa !9
+float*8B

	full_text


float* %42
6fadd8B,
*
	full_text

%44 = fadd float %43, %38
)float8B

	full_text

	float %43
)float8B

	full_text

	float %38
Lstore8BA
?
	full_text2
0
.store float %44, float* %12, align 4, !tbaa !9
)float8B

	full_text

	float %44
+float*8B

	full_text


float* %12
.or8B&
$
	full_text

%45 = or i64 %26, 3
%i648B

	full_text
	
i64 %26
6mul8B-
+
	full_text

%46 = mul nsw i64 %45, %15
%i648B

	full_text
	
i64 %45
%i648B

	full_text
	
i64 %15
6add8B-
+
	full_text

%47 = add nsw i64 %46, %17
%i648B

	full_text
	
i64 %46
%i648B

	full_text
	
i64 %17
\getelementptr8BI
G
	full_text:
8
6%48 = getelementptr inbounds float, float* %1, i64 %47
%i648B

	full_text
	
i64 %47
Lload8BB
@
	full_text3
1
/%49 = load float, float* %48, align 4, !tbaa !9
+float*8B

	full_text


float* %48
6fadd8B,
*
	full_text

%50 = fadd float %49, %44
)float8B

	full_text

	float %49
)float8B

	full_text

	float %44
Lstore8BA
?
	full_text2
0
.store float %50, float* %12, align 4, !tbaa !9
)float8B

	full_text

	float %50
+float*8B

	full_text


float* %12
4add8B+
)
	full_text

%51 = add nsw i64 %26, 4
%i648B

	full_text
	
i64 %26
1add8B(
&
	full_text

%52 = add i64 %27, -4
%i648B

	full_text
	
i64 %27
5icmp8B+
)
	full_text

%53 = icmp eq i64 %52, 0
%i648B

	full_text
	
i64 %52
:br8B2
0
	full_text#
!
br i1 %53, label %54, label %24
#i18B

	full_text


i1 %53
Hphi8B?
=
	full_text0
.
,%55 = phi float [ undef, %14 ], [ %50, %24 ]
)float8B

	full_text

	float %50
Ophi8BF
D
	full_text7
5
3%56 = phi float [ 0.000000e+00, %14 ], [ %50, %24 ]
)float8B

	full_text

	float %50
Bphi8B9
7
	full_text*
(
&%57 = phi i64 [ 0, %14 ], [ %51, %24 ]
%i648B

	full_text
	
i64 %51
5icmp8B+
)
	full_text

%58 = icmp eq i64 %20, 0
%i648B

	full_text
	
i64 %20
:br8B2
0
	full_text#
!
br i1 %58, label %72, label %59
#i18B

	full_text


i1 %58
'br8B

	full_text

br label %60
Fphi8B=
;
	full_text.
,
*%61 = phi float [ %56, %59 ], [ %68, %60 ]
)float8B

	full_text

	float %56
)float8B

	full_text

	float %68
Dphi8B;
9
	full_text,
*
(%62 = phi i64 [ %57, %59 ], [ %69, %60 ]
%i648B

	full_text
	
i64 %57
%i648B

	full_text
	
i64 %69
Dphi8B;
9
	full_text,
*
(%63 = phi i64 [ %20, %59 ], [ %70, %60 ]
%i648B

	full_text
	
i64 %20
%i648B

	full_text
	
i64 %70
6mul8B-
+
	full_text

%64 = mul nsw i64 %62, %15
%i648B

	full_text
	
i64 %62
%i648B

	full_text
	
i64 %15
6add8B-
+
	full_text

%65 = add nsw i64 %64, %17
%i648B

	full_text
	
i64 %64
%i648B

	full_text
	
i64 %17
\getelementptr8BI
G
	full_text:
8
6%66 = getelementptr inbounds float, float* %1, i64 %65
%i648B

	full_text
	
i64 %65
Lload8BB
@
	full_text3
1
/%67 = load float, float* %66, align 4, !tbaa !9
+float*8B

	full_text


float* %66
6fadd8B,
*
	full_text

%68 = fadd float %67, %61
)float8B

	full_text

	float %67
)float8B

	full_text

	float %61
Lstore8BA
?
	full_text2
0
.store float %68, float* %12, align 4, !tbaa !9
)float8B

	full_text

	float %68
+float*8B

	full_text


float* %12
8add8B/
-
	full_text 

%69 = add nuw nsw i64 %62, 1
%i648B

	full_text
	
i64 %62
1add8B(
&
	full_text

%70 = add i64 %63, -1
%i648B

	full_text
	
i64 %63
5icmp8B+
)
	full_text

%71 = icmp eq i64 %70, 0
%i648B

	full_text
	
i64 %70
Jbr8BB
@
	full_text3
1
/br i1 %71, label %72, label %60, !llvm.loop !13
#i18B

	full_text


i1 %71
\phi8BS
Q
	full_textD
B
@%73 = phi float [ 0.000000e+00, %9 ], [ %55, %54 ], [ %68, %60 ]
)float8B

	full_text

	float %55
)float8B

	full_text

	float %68
Bfdiv8B8
6
	full_text)
'
%%74 = fdiv float %73, %2, !fpmath !15
)float8B

	full_text

	float %73
Lstore8BA
?
	full_text2
0
.store float %74, float* %12, align 4, !tbaa !9
)float8B

	full_text

	float %74
+float*8B

	full_text


float* %12
'br8B

	full_text

br label %75
$ret8	B

	full_text


ret void
*float*8
B

	full_text

	float* %0
$i328
B

	full_text


i32 %3
*float*8
B

	full_text

	float* %1
(float8
B

	full_text


float %2
$i328
B

	full_text


i32 %4
-; undefined function B

	full_text

 
#i648
B

	full_text	

i64 2
#i648
B

	full_text	

i64 4
#i328
B

	full_text	

i32 0
2float8
B%
#
	full_text

float 0.000000e+00
#i648
B

	full_text	

i64 0
$i648
B

	full_text


i64 -1
+float8
B

	full_text

float undef
#i648
B

	full_text	

i64 3
$i648
B

	full_text


i64 32
$i648
B

	full_text


i64 -4
#i648
B

	full_text	

i64 1      	  
 

                     " !# !! $& %% '( '' )* )+ )) ,- ,. ,, /0 /1 // 23 22 45 44 67 68 66 9: 9; 99 <= << >? >@ >> AB AC AA DE DD FG FF HI HJ HH KL KM KK NO NN PQ PR PP ST SU SS VW VV XY XX Z[ Z\ ZZ ]^ ]_ ]] `a `` bc bd bb ef eg ee hi hh jk jj lm ln ll op oq oo rs rr tu tt vw vv xy x{ zz |} || ~ ~~ ÄÅ ÄÄ ÇÉ ÇÜ Ö
á ÖÖ àâ à
ä àà ãå ã
ç ãã éè é
ê éé ëí ë
ì ëë î
ï îî ñó ññ òô ò
ö òò õú õ
ù õõ ûü ûû †° †† ¢£ ¢¢ §• §
ß ¶
® ¶¶ ©™ ©© ´¨ ´
≠ ´´ Æ∞ 	± ± ≤ 2≤ D≤ V≤ h≤ î
≥ ©¥ ¥     	 
           " #l &r (! *t +' - ., 0 1/ 32 54 7% 86 : ;' =< ? @> B CA ED GF I6 JH L M' ON Q RP T US WV YX [H \Z ^ _' a` c db f ge ih kj mZ nl p q' s) ut wv yl {l }r  ÅÄ É| Üò á~ âû ä å† çà è êé í ìë ïî óñ ôÖ öò ú ùà üã °† £¢ •z ßò ®¶ ™© ¨ ≠  Ø  ¶ z !Æ ØÇ ¶Ç Ñ$ %Ñ Öx zx %§ ¶§ Ö Ø µµ µµ 	∂ N	∑ r∏ 	∏ π π %π |π ¶∫ '	∫ v∫ ~
∫ Ä
∫ ¢	ª 
ª †º z	Ω 	Ω 	Ω `	æ 	æ 
	æ 	æ 	ø t	¿ <
¿ û"
mean_kernel"
_Z13get_global_idj*§
+polybench-gpu-1.0-covariance-mean_kernel.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02Å

devmap_label
 
 
transfer_bytes_log1p
££äA

wgsize
Ä

transfer_bytes
å¿Ç

wgsize_log1p
££äA