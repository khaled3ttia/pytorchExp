

[external]
KcallBC
A
	full_text4
2
0%7 = tail call i64 @_Z13get_global_idj(i32 0) #3
0addB)
'
	full_text

%8 = add nsw i32 %3, 1
2zextB*
(
	full_text

%9 = zext i32 %8 to i64
"i32B

	full_text


i32 %8
.addB'
%
	full_text

%10 = add i64 %7, %9
"i64B

	full_text


i64 %7
"i64B

	full_text


i64 %9
6truncB-
+
	full_text

%11 = trunc i64 %10 to i32
#i64B

	full_text
	
i64 %10
5icmpB-
+
	full_text

%12 = icmp slt i32 %11, %5
#i32B

	full_text
	
i32 %11
9brB3
1
	full_text$
"
 br i1 %12, label %13, label %131
!i1B

	full_text


i1 %12
4mul8B+
)
	full_text

%14 = mul nsw i32 %5, %3
6add8B-
+
	full_text

%15 = add nsw i32 %14, %11
%i328B

	full_text
	
i32 %14
%i328B

	full_text
	
i32 %11
6sext8B,
*
	full_text

%16 = sext i32 %15 to i64
%i328B

	full_text
	
i32 %15
\getelementptr8BI
G
	full_text:
8
6%17 = getelementptr inbounds float, float* %1, i64 %16
%i648B

	full_text
	
i64 %16
Ustore8BJ
H
	full_text;
9
7store float 0.000000e+00, float* %17, align 4, !tbaa !9
+float*8B

	full_text


float* %17
5icmp8B+
)
	full_text

%18 = icmp sgt i32 %4, 0
;br8B3
1
	full_text$
"
 br i1 %18, label %19, label %131
#i18B

	full_text


i1 %18
5sext8B+
)
	full_text

%20 = sext i32 %5 to i64
5sext8B+
)
	full_text

%21 = sext i32 %3 to i64
1shl8B(
&
	full_text

%22 = shl i64 %10, 32
%i648B

	full_text
	
i64 %10
9ashr8B/
-
	full_text 

%23 = ashr exact i64 %22, 32
%i648B

	full_text
	
i64 %22
5zext8B+
)
	full_text

%24 = zext i32 %4 to i64
0and8B'
%
	full_text

%25 = and i64 %24, 1
%i648B

	full_text
	
i64 %24
4icmp8B*
(
	full_text

%26 = icmp eq i32 %4, 1
:br8B2
0
	full_text#
!
br i1 %26, label %53, label %27
#i18B

	full_text


i1 %26
6sub8B-
+
	full_text

%28 = sub nsw i64 %24, %25
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %25
'br8B

	full_text

br label %29
Ophi8BF
D
	full_text7
5
3%30 = phi float [ 0.000000e+00, %27 ], [ %49, %29 ]
)float8B

	full_text

	float %49
Bphi8B9
7
	full_text*
(
&%31 = phi i64 [ 0, %27 ], [ %50, %29 ]
%i648B

	full_text
	
i64 %50
Dphi8B;
9
	full_text,
*
(%32 = phi i64 [ %28, %27 ], [ %51, %29 ]
%i648B

	full_text
	
i64 %28
%i648B

	full_text
	
i64 %51
6mul8B-
+
	full_text

%33 = mul nsw i64 %31, %20
%i648B

	full_text
	
i64 %31
%i648B

	full_text
	
i64 %20
6add8B-
+
	full_text

%34 = add nsw i64 %33, %21
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %21
\getelementptr8BI
G
	full_text:
8
6%35 = getelementptr inbounds float, float* %2, i64 %34
%i648B

	full_text
	
i64 %34
Lload8BB
@
	full_text3
1
/%36 = load float, float* %35, align 4, !tbaa !9
+float*8B

	full_text


float* %35
6add8B-
+
	full_text

%37 = add nsw i64 %33, %23
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %23
\getelementptr8BI
G
	full_text:
8
6%38 = getelementptr inbounds float, float* %0, i64 %37
%i648B

	full_text
	
i64 %37
Lload8BB
@
	full_text3
1
/%39 = load float, float* %38, align 4, !tbaa !9
+float*8B

	full_text


float* %38
ecall8B[
Y
	full_textL
J
H%40 = tail call float @llvm.fmuladd.f32(float %36, float %39, float %30)
)float8B

	full_text

	float %36
)float8B

	full_text

	float %39
)float8B

	full_text

	float %30
Lstore8BA
?
	full_text2
0
.store float %40, float* %17, align 4, !tbaa !9
)float8B

	full_text

	float %40
+float*8B

	full_text


float* %17
.or8B&
$
	full_text

%41 = or i64 %31, 1
%i648B

	full_text
	
i64 %31
6mul8B-
+
	full_text

%42 = mul nsw i64 %41, %20
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %20
6add8B-
+
	full_text

%43 = add nsw i64 %42, %21
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %21
\getelementptr8BI
G
	full_text:
8
6%44 = getelementptr inbounds float, float* %2, i64 %43
%i648B

	full_text
	
i64 %43
Lload8BB
@
	full_text3
1
/%45 = load float, float* %44, align 4, !tbaa !9
+float*8B

	full_text


float* %44
6add8B-
+
	full_text

%46 = add nsw i64 %42, %23
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %23
\getelementptr8BI
G
	full_text:
8
6%47 = getelementptr inbounds float, float* %0, i64 %46
%i648B

	full_text
	
i64 %46
Lload8BB
@
	full_text3
1
/%48 = load float, float* %47, align 4, !tbaa !9
+float*8B

	full_text


float* %47
ecall8B[
Y
	full_textL
J
H%49 = tail call float @llvm.fmuladd.f32(float %45, float %48, float %40)
)float8B

	full_text

	float %45
)float8B

	full_text

	float %48
)float8B

	full_text

	float %40
Lstore8BA
?
	full_text2
0
.store float %49, float* %17, align 4, !tbaa !9
)float8B

	full_text

	float %49
+float*8B

	full_text


float* %17
4add8B+
)
	full_text

%50 = add nsw i64 %31, 2
%i648B

	full_text
	
i64 %31
1add8B(
&
	full_text

%51 = add i64 %32, -2
%i648B

	full_text
	
i64 %32
5icmp8B+
)
	full_text

%52 = icmp eq i64 %51, 0
%i648B

	full_text
	
i64 %51
:br8B2
0
	full_text#
!
br i1 %52, label %53, label %29
#i18B

	full_text


i1 %52
Hphi8B?
=
	full_text0
.
,%54 = phi float [ undef, %19 ], [ %49, %29 ]
)float8B

	full_text

	float %49
Ophi8BF
D
	full_text7
5
3%55 = phi float [ 0.000000e+00, %19 ], [ %49, %29 ]
)float8B

	full_text

	float %49
Bphi8B9
7
	full_text*
(
&%56 = phi i64 [ 0, %19 ], [ %50, %29 ]
%i648B

	full_text
	
i64 %50
5icmp8B+
)
	full_text

%57 = icmp eq i64 %25, 0
%i648B

	full_text
	
i64 %25
:br8B2
0
	full_text#
!
br i1 %57, label %67, label %58
#i18B

	full_text


i1 %57
6mul8B-
+
	full_text

%59 = mul nsw i64 %56, %20
%i648B

	full_text
	
i64 %56
%i648B

	full_text
	
i64 %20
6add8B-
+
	full_text

%60 = add nsw i64 %59, %21
%i648B

	full_text
	
i64 %59
%i648B

	full_text
	
i64 %21
\getelementptr8BI
G
	full_text:
8
6%61 = getelementptr inbounds float, float* %2, i64 %60
%i648B

	full_text
	
i64 %60
Lload8BB
@
	full_text3
1
/%62 = load float, float* %61, align 4, !tbaa !9
+float*8B

	full_text


float* %61
6add8B-
+
	full_text

%63 = add nsw i64 %59, %23
%i648B

	full_text
	
i64 %59
%i648B

	full_text
	
i64 %23
\getelementptr8BI
G
	full_text:
8
6%64 = getelementptr inbounds float, float* %0, i64 %63
%i648B

	full_text
	
i64 %63
Lload8BB
@
	full_text3
1
/%65 = load float, float* %64, align 4, !tbaa !9
+float*8B

	full_text


float* %64
ecall8B[
Y
	full_textL
J
H%66 = tail call float @llvm.fmuladd.f32(float %62, float %65, float %55)
)float8B

	full_text

	float %62
)float8B

	full_text

	float %65
)float8B

	full_text

	float %55
Lstore8BA
?
	full_text2
0
.store float %66, float* %17, align 4, !tbaa !9
)float8B

	full_text

	float %66
+float*8B

	full_text


float* %17
'br8B

	full_text

br label %67
Fphi8B=
;
	full_text.
,
*%68 = phi float [ %54, %53 ], [ %66, %58 ]
)float8B

	full_text

	float %54
)float8B

	full_text

	float %66
;br8B3
1
	full_text$
"
 br i1 %18, label %69, label %131
#i18B

	full_text


i1 %18
5sext8B+
)
	full_text

%70 = sext i32 %5 to i64
5sext8B+
)
	full_text

%71 = sext i32 %3 to i64
1shl8B(
&
	full_text

%72 = shl i64 %10, 32
%i648B

	full_text
	
i64 %10
9ashr8B/
-
	full_text 

%73 = ashr exact i64 %72, 32
%i648B

	full_text
	
i64 %72
5zext8B+
)
	full_text

%74 = zext i32 %4 to i64
\getelementptr8BI
G
	full_text:
8
6%75 = getelementptr inbounds float, float* %2, i64 %71
%i648B

	full_text
	
i64 %71
Lload8BB
@
	full_text3
1
/%76 = load float, float* %75, align 4, !tbaa !9
+float*8B

	full_text


float* %75
\getelementptr8BI
G
	full_text:
8
6%77 = getelementptr inbounds float, float* %0, i64 %73
%i648B

	full_text
	
i64 %73
Lload8BB
@
	full_text3
1
/%78 = load float, float* %77, align 4, !tbaa !9
+float*8B

	full_text


float* %77
@fsub8B6
4
	full_text'
%
#%79 = fsub float -0.000000e+00, %76
)float8B

	full_text

	float %76
ecall8B[
Y
	full_textL
J
H%80 = tail call float @llvm.fmuladd.f32(float %79, float %68, float %78)
)float8B

	full_text

	float %79
)float8B

	full_text

	float %68
)float8B

	full_text

	float %78
Lstore8BA
?
	full_text2
0
.store float %80, float* %77, align 4, !tbaa !9
)float8B

	full_text

	float %80
+float*8B

	full_text


float* %77
4icmp8B*
(
	full_text

%81 = icmp eq i32 %4, 1
;br8B3
1
	full_text$
"
 br i1 %81, label %131, label %82
#i18B

	full_text


i1 %81
/and8	B&
$
	full_text

%83 = and i32 %4, 1
0xor8	B'
%
	full_text

%84 = xor i32 %83, 1
%i328	B

	full_text
	
i32 %83
4icmp8	B*
(
	full_text

%85 = icmp eq i32 %4, 2
;br8	B3
1
	full_text$
"
 br i1 %85, label %117, label %86
#i18	B

	full_text


i1 %85
6zext8
B,
*
	full_text

%87 = zext i32 %84 to i64
%i328
B

	full_text
	
i32 %84
5add8
B,
*
	full_text

%88 = add nsw i64 %74, -1
%i648
B

	full_text
	
i64 %74
6sub8
B-
+
	full_text

%89 = sub nsw i64 %88, %87
%i648
B

	full_text
	
i64 %88
%i648
B

	full_text
	
i64 %87
'br8
B

	full_text

br label %90
Cphi8B:
8
	full_text+
)
'%91 = phi i64 [ 1, %86 ], [ %114, %90 ]
&i648B

	full_text


i64 %114
Ephi8B<
:
	full_text-
+
)%92 = phi i64 [ %89, %86 ], [ %115, %90 ]
%i648B

	full_text
	
i64 %89
&i648B

	full_text


i64 %115
Lload8BB
@
	full_text3
1
/%93 = load float, float* %17, align 4, !tbaa !9
+float*8B

	full_text


float* %17
6mul8B-
+
	full_text

%94 = mul nsw i64 %91, %70
%i648B

	full_text
	
i64 %91
%i648B

	full_text
	
i64 %70
6add8B-
+
	full_text

%95 = add nsw i64 %94, %71
%i648B

	full_text
	
i64 %94
%i648B

	full_text
	
i64 %71
\getelementptr8BI
G
	full_text:
8
6%96 = getelementptr inbounds float, float* %2, i64 %95
%i648B

	full_text
	
i64 %95
Lload8BB
@
	full_text3
1
/%97 = load float, float* %96, align 4, !tbaa !9
+float*8B

	full_text


float* %96
6add8B-
+
	full_text

%98 = add nsw i64 %94, %73
%i648B

	full_text
	
i64 %94
%i648B

	full_text
	
i64 %73
\getelementptr8BI
G
	full_text:
8
6%99 = getelementptr inbounds float, float* %0, i64 %98
%i648B

	full_text
	
i64 %98
Mload8BC
A
	full_text4
2
0%100 = load float, float* %99, align 4, !tbaa !9
+float*8B

	full_text


float* %99
Afsub8B7
5
	full_text(
&
$%101 = fsub float -0.000000e+00, %97
)float8B

	full_text

	float %97
hcall8B^
\
	full_textO
M
K%102 = tail call float @llvm.fmuladd.f32(float %101, float %93, float %100)
*float8B

	full_text


float %101
)float8B

	full_text

	float %93
*float8B

	full_text


float %100
Mstore8BB
@
	full_text3
1
/store float %102, float* %99, align 4, !tbaa !9
*float8B

	full_text


float %102
+float*8B

	full_text


float* %99
9add8B0
.
	full_text!

%103 = add nuw nsw i64 %91, 1
%i648B

	full_text
	
i64 %91
Mload8BC
A
	full_text4
2
0%104 = load float, float* %17, align 4, !tbaa !9
+float*8B

	full_text


float* %17
8mul8B/
-
	full_text 

%105 = mul nsw i64 %103, %70
&i648B

	full_text


i64 %103
%i648B

	full_text
	
i64 %70
8add8B/
-
	full_text 

%106 = add nsw i64 %105, %71
&i648B

	full_text


i64 %105
%i648B

	full_text
	
i64 %71
^getelementptr8BK
I
	full_text<
:
8%107 = getelementptr inbounds float, float* %2, i64 %106
&i648B

	full_text


i64 %106
Nload8BD
B
	full_text5
3
1%108 = load float, float* %107, align 4, !tbaa !9
,float*8B

	full_text

float* %107
8add8B/
-
	full_text 

%109 = add nsw i64 %105, %73
&i648B

	full_text


i64 %105
%i648B

	full_text
	
i64 %73
^getelementptr8BK
I
	full_text<
:
8%110 = getelementptr inbounds float, float* %0, i64 %109
&i648B

	full_text


i64 %109
Nload8BD
B
	full_text5
3
1%111 = load float, float* %110, align 4, !tbaa !9
,float*8B

	full_text

float* %110
Bfsub8B8
6
	full_text)
'
%%112 = fsub float -0.000000e+00, %108
*float8B

	full_text


float %108
icall8B_
]
	full_textP
N
L%113 = tail call float @llvm.fmuladd.f32(float %112, float %104, float %111)
*float8B

	full_text


float %112
*float8B

	full_text


float %104
*float8B

	full_text


float %111
Nstore8BC
A
	full_text4
2
0store float %113, float* %110, align 4, !tbaa !9
*float8B

	full_text


float %113
,float*8B

	full_text

float* %110
5add8B,
*
	full_text

%114 = add nsw i64 %91, 2
%i648B

	full_text
	
i64 %91
2add8B)
'
	full_text

%115 = add i64 %92, -2
%i648B

	full_text
	
i64 %92
7icmp8B-
+
	full_text

%116 = icmp eq i64 %115, 0
&i648B

	full_text


i64 %115
<br8B4
2
	full_text%
#
!br i1 %116, label %117, label %90
$i18B

	full_text
	
i1 %116
Dphi8B;
9
	full_text,
*
(%118 = phi i64 [ 1, %82 ], [ %114, %90 ]
&i648B

	full_text


i64 %114
6icmp8B,
*
	full_text

%119 = icmp eq i32 %84, 0
%i328B

	full_text
	
i32 %84
=br8B5
3
	full_text&
$
"br i1 %119, label %131, label %120
$i18B

	full_text
	
i1 %119
Mload8BC
A
	full_text4
2
0%121 = load float, float* %17, align 4, !tbaa !9
+float*8B

	full_text


float* %17
8mul8B/
-
	full_text 

%122 = mul nsw i64 %118, %70
&i648B

	full_text


i64 %118
%i648B

	full_text
	
i64 %70
8add8B/
-
	full_text 

%123 = add nsw i64 %122, %71
&i648B

	full_text


i64 %122
%i648B

	full_text
	
i64 %71
^getelementptr8BK
I
	full_text<
:
8%124 = getelementptr inbounds float, float* %2, i64 %123
&i648B

	full_text


i64 %123
Nload8BD
B
	full_text5
3
1%125 = load float, float* %124, align 4, !tbaa !9
,float*8B

	full_text

float* %124
8add8B/
-
	full_text 

%126 = add nsw i64 %122, %73
&i648B

	full_text


i64 %122
%i648B

	full_text
	
i64 %73
^getelementptr8BK
I
	full_text<
:
8%127 = getelementptr inbounds float, float* %0, i64 %126
&i648B

	full_text


i64 %126
Nload8BD
B
	full_text5
3
1%128 = load float, float* %127, align 4, !tbaa !9
,float*8B

	full_text

float* %127
Bfsub8B8
6
	full_text)
'
%%129 = fsub float -0.000000e+00, %125
*float8B

	full_text


float %125
icall8B_
]
	full_textP
N
L%130 = tail call float @llvm.fmuladd.f32(float %129, float %121, float %128)
*float8B

	full_text


float %129
*float8B

	full_text


float %121
*float8B

	full_text


float %128
Nstore8BC
A
	full_text4
2
0store float %130, float* %127, align 4, !tbaa !9
*float8B

	full_text


float %130
,float*8B

	full_text

float* %127
(br8B 

	full_text

br label %131
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %4
*float*8B

	full_text

	float* %1
*float*8B

	full_text

	float* %2
$i328B

	full_text


i32 %5
$i328B

	full_text


i32 %3
*float*8B

	full_text

	float* %0
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
2float8B%
#
	full_text

float 0.000000e+00
#i328B

	full_text	

i32 1
$i648B

	full_text


i64 32
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 2
$i648B

	full_text


i64 -2
#i648B

	full_text	

i64 1
+float8B

	full_text

float undef
#i648B

	full_text	

i64 0
3float8B&
$
	full_text

float -0.000000e+00
#i328B

	full_text	

i32 2
$i648B

	full_text


i64 -1       	  
 

                     !! "# "" $$ %& %( ') '' *, ++ -. -- /0 /1 // 23 24 22 56 57 55 89 88 :; :: <= <> << ?@ ?? AB AA CD CE CF CC GH GI GG JK JJ LM LN LL OP OQ OO RS RR TU TT VW VX VV YZ YY [\ [[ ]^ ]_ ]` ]] ab ac aa de dd fg ff hi hh jk jm ll no nn pq pp rs rr tu tw vx vv yz y{ yy |} || ~ ~~ € €
‚ €€ ƒ
„ ƒƒ …† …… ‡ˆ ‡
‰ ‡
Š ‡‡ ‹Œ ‹
 ‹‹ Ž 
‘  ’“ ’” •• –— –– ˜™ ˜˜ šš ›
œ ›› ž  Ÿ
  ŸŸ ¡¢ ¡¡ £
¤ ££ ¥¦ ¥
§ ¥
¨ ¥¥ ©ª ©
« ©© ¬¬ ­® ­¯ °± °° ²² ³´ ³¶ µµ ·¸ ·· ¹º ¹
» ¹¹ ¼
¾ ½½ ¿À ¿
Á ¿¿ ÂÃ ÂÂ ÄÅ Ä
Æ ÄÄ ÇÈ Ç
É ÇÇ Ê
Ë ÊÊ ÌÍ ÌÌ ÎÏ Î
Ð ÎÎ Ñ
Ò ÑÑ ÓÔ ÓÓ Õ
Ö ÕÕ ×Ø ×
Ù ×
Ú ×× ÛÜ Û
Ý ÛÛ Þß ÞÞ àá àà âã â
ä ââ åæ å
ç åå è
é èè êë êê ìí ì
î ìì ï
ð ïï ñò ññ ó
ô óó õö õ
÷ õ
ø õõ ùú ù
û ùù üý üü þÿ þþ € €€ ‚ƒ ‚
… „„ †‡ †† ˆ‰ ˆ‹ ŠŠ Œ Œ
Ž ŒŒ  
‘  ’
“ ’’ ”• ”” –— –
˜ –– ™
š ™™ ›œ ›› 
ž  Ÿ  Ÿ
¡ Ÿ
¢ ŸŸ £¤ £
¥ ££ ¦¨ ¨ !¨ $¨ š¨ ¬¨ ¯¨ ²© ª 8ª Rª |ª ›ª Êª èª ’	« 
« « « ”¬ 	¬ ¬ ¬ •­ ?­ Y­ ƒ­ Ÿ­ Ñ­ ï­ ™    	 
          ! #$ &! (" )] ,d .' 0f 1- 3 42 6 75 98 ;2 = >< @? B: DA E+ FC H I- KJ M NL P QO SR UL W XV ZY \T ^[ _C `] b c- e/ gf ih k] m] od q" sr up w xv z {y }| v  ‚€ „ƒ †~ ˆ… ‰n Š‡ Œ l ‡ ‘ “ —– ™• œ› ž˜  Ÿ ¢ ¤£ ¦ §¡ ¨¥ ªŸ «¬ ®¯ ±² ´° ¶š ¸· ºµ »ü ¾¹ Àþ Á Ã½ Å” ÆÄ È• ÉÇ ËÊ ÍÄ Ï˜ ÐÎ ÒÑ ÔÌ ÖÕ ØÂ ÙÓ Ú× ÜÑ Ý½ ß áÞ ã” äâ æ• çå éè ëâ í˜ îì ðï òê ôó öà ÷ñ øõ úï û½ ý¿ ÿþ € ƒü …° ‡† ‰ ‹„ ” ŽŒ • ‘ “’ •Œ —˜ ˜– š™ œ” ž  Š ¡› ¢Ÿ ¤™ ¥  §  §% l% 't t v* +’ ”’ §Ž j lj +­ §­ ¯³ „³ µˆ §ˆ Š¼ ½¦ §‚ „‚ ½ § ®® ¯¯‡ ¯¯ ‡Ÿ ¯¯ Ÿ ®® ] ¯¯ ]× ¯¯ ×C ¯¯ C¥ ¯¯ ¥õ ¯¯ õ° ° +° n	± 	± $
± ¬
± ¯
± °	² 	² 
² –
² ˜³ 	³ 
³ †	´ d
´ ü	µ f
µ þ	¶ "	¶ J¶ ½
¶ Þ¶ „· l¸ -	¸ h¸ p	¸ r
¸ €¹ £¹ Õ¹ ó¹ 
º ²
» ·"
gramschmidt_kernel3"
_Z13get_global_idj"
llvm.fmuladd.f32*­
4polybench-gpu-1.0-gramschmidt-gramschmidt_kernel3.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02

devmap_label


wgsize
€
 
transfer_bytes_log1p
k“A

transfer_bytes
€€€0

wgsize_log1p
k“A