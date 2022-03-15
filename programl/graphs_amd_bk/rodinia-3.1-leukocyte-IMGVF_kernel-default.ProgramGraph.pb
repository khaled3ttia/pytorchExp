

[external]
KcallBC
A
	full_text4
2
0%11 = tail call i64 @_Z12get_group_idj(i32 0) #5
/shlB(
&
	full_text

%12 = shl i64 %11, 32
#i64B

	full_text
	
i64 %11
7ashrB/
-
	full_text 

%13 = ashr exact i64 %12, 32
#i64B

	full_text
	
i64 %12
VgetelementptrBE
C
	full_text6
4
2%14 = getelementptr inbounds i32, i32* %2, i64 %13
#i64B

	full_text
	
i64 %13
FloadB>
<
	full_text/
-
+%15 = load i32, i32* %14, align 4, !tbaa !8
%i32*B

	full_text


i32* %14
4sextB,
*
	full_text

%16 = sext i32 %15 to i64
#i32B

	full_text
	
i32 %15
ZgetelementptrBI
G
	full_text:
8
6%17 = getelementptr inbounds float, float* %0, i64 %16
#i64B

	full_text
	
i64 %16
ZgetelementptrBI
G
	full_text:
8
6%18 = getelementptr inbounds float, float* %1, i64 %16
#i64B

	full_text
	
i64 %16
VgetelementptrBE
C
	full_text6
4
2%19 = getelementptr inbounds i32, i32* %3, i64 %13
#i64B

	full_text
	
i64 %13
FloadB>
<
	full_text/
-
+%20 = load i32, i32* %19, align 4, !tbaa !8
%i32*B

	full_text


i32* %19
VgetelementptrBE
C
	full_text6
4
2%21 = getelementptr inbounds i32, i32* %4, i64 %13
#i64B

	full_text
	
i64 %13
FloadB>
<
	full_text/
-
+%22 = load i32, i32* %21, align 4, !tbaa !8
%i32*B

	full_text


i32* %21
4mulB-
+
	full_text

%23 = mul nsw i32 %22, %20
#i32B

	full_text
	
i32 %22
#i32B

	full_text
	
i32 %20
4addB-
+
	full_text

%24 = add nsw i32 %23, 255
#i32B

	full_text
	
i32 %23
2sdivB*
(
	full_text

%25 = sdiv i32 %24, 256
#i32B

	full_text
	
i32 %24
KcallBC
A
	full_text4
2
0%26 = tail call i64 @_Z12get_local_idj(i32 0) #5
6truncB-
+
	full_text

%27 = trunc i64 %26 to i32
#i64B

	full_text
	
i64 %26
4icmpB,
*
	full_text

%28 = icmp sgt i32 %23, 0
#i32B

	full_text
	
i32 %23
8brB2
0
	full_text#
!
br i1 %28, label %29, label %49
!i1B

	full_text


i1 %28
'br8B

	full_text

br label %30
Bphi8B9
7
	full_text*
(
&%31 = phi i32 [ %47, %46 ], [ 0, %29 ]
%i328B

	full_text
	
i32 %47
4shl8B+
)
	full_text

%32 = shl nsw i32 %31, 8
%i328B

	full_text
	
i32 %31
6add8B-
+
	full_text

%33 = add nsw i32 %32, %27
%i328B

	full_text
	
i32 %32
%i328B

	full_text
	
i32 %27
4sdiv8B*
(
	full_text

%34 = sdiv i32 %33, %22
%i328B

	full_text
	
i32 %33
%i328B

	full_text
	
i32 %22
8icmp8B.
,
	full_text

%35 = icmp slt i32 %34, %20
%i328B

	full_text
	
i32 %34
%i328B

	full_text
	
i32 %20
:br8B2
0
	full_text#
!
br i1 %35, label %36, label %46
#i18B

	full_text


i1 %35
4srem8B*
(
	full_text

%37 = srem i32 %33, %22
%i328B

	full_text
	
i32 %33
%i328B

	full_text
	
i32 %22
6mul8B-
+
	full_text

%38 = mul nsw i32 %34, %22
%i328B

	full_text
	
i32 %34
%i328B

	full_text
	
i32 %22
6add8B-
+
	full_text

%39 = add nsw i32 %37, %38
%i328B

	full_text
	
i32 %37
%i328B

	full_text
	
i32 %38
6sext8B,
*
	full_text

%40 = sext i32 %39 to i64
%i328B

	full_text
	
i32 %39
]getelementptr8BJ
H
	full_text;
9
7%41 = getelementptr inbounds float, float* %17, i64 %40
+float*8B

	full_text


float* %17
%i648B

	full_text
	
i64 %40
@bitcast8B3
1
	full_text$
"
 %42 = bitcast float* %41 to i32*
+float*8B

	full_text


float* %41
Iload8B?
=
	full_text0
.
,%43 = load i32, i32* %42, align 4, !tbaa !12
'i32*8B

	full_text


i32* %42
†getelementptr8Bs
q
	full_textd
b
`%44 = getelementptr inbounds [3321 x float], [3321 x float]* @IMGVF_kernel.IMGVF, i64 0, i64 %40
%i648B

	full_text
	
i64 %40
@bitcast8B3
1
	full_text$
"
 %45 = bitcast float* %44 to i32*
+float*8B

	full_text


float* %44
Istore8B>
<
	full_text/
-
+store i32 %43, i32* %45, align 4, !tbaa !12
%i328B

	full_text
	
i32 %43
'i32*8B

	full_text


i32* %45
'br8B

	full_text

br label %46
8add8B/
-
	full_text 

%47 = add nuw nsw i32 %31, 1
%i328B

	full_text
	
i32 %31
8icmp8B.
,
	full_text

%48 = icmp slt i32 %47, %25
%i328B

	full_text
	
i32 %47
%i328B

	full_text
	
i32 %25
:br8B2
0
	full_text#
!
br i1 %48, label %30, label %49
#i18B

	full_text


i1 %48
Fphi8B=
;
	full_text.
,
*%50 = phi i32 [ undef, %10 ], [ %34, %46 ]
%i328B

	full_text
	
i32 %34
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #6
5icmp8B+
)
	full_text

%51 = icmp eq i32 %27, 0
%i328B

	full_text
	
i32 %27
:br8B2
0
	full_text#
!
br i1 %51, label %52, label %53
#i18B

	full_text


i1 %51
_store8BT
R
	full_textE
C
Astore i32 0, i32* @IMGVF_kernel.cell_converged, align 4, !tbaa !8
'br8B

	full_text

br label %53
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #6
<sitofp8B0
.
	full_text!

%54 = sitofp i32 %22 to float
%i328B

	full_text
	
i32 %22
Lfdiv8BB
@
	full_text3
1
/%55 = fdiv float 1.000000e+00, %54, !fpmath !14
)float8B

	full_text

	float %54
4srem8B*
(
	full_text

%56 = srem i32 256, %22
%i328B

	full_text
	
i32 %22
Kfdiv8BA
?
	full_text2
0
.%57 = fdiv float 1.000000e+00, %7, !fpmath !14
aload8BW
U
	full_textH
F
D%58 = load i32, i32* @IMGVF_kernel.cell_converged, align 4, !tbaa !8
5icmp8B+
)
	full_text

%59 = icmp eq i32 %58, 0
%i328B

	full_text
	
i32 %58
5icmp8B+
)
	full_text

%60 = icmp sgt i32 %8, 0
1and8B(
&
	full_text

%61 = and i1 %60, %59
#i18B

	full_text


i1 %60
#i18B

	full_text


i1 %59
;br8B3
1
	full_text$
"
 br i1 %61, label %62, label %294
#i18B

	full_text


i1 %61
4srem8B*
(
	full_text

%63 = srem i32 %27, %22
%i328B

	full_text
	
i32 %27
%i328B

	full_text
	
i32 %22
6sub8B-
+
	full_text

%64 = sub nsw i32 %63, %56
%i328B

	full_text
	
i32 %63
%i328B

	full_text
	
i32 %56
1shl8B(
&
	full_text

%65 = shl i64 %26, 32
%i648B

	full_text
	
i64 %26
9ashr8B/
-
	full_text 

%66 = ashr exact i64 %65, 32
%i648B

	full_text
	
i64 %65
…getelementptr8Br
p
	full_textc
a
_%67 = getelementptr inbounds [256 x float], [256 x float]* @IMGVF_kernel.buffer, i64 0, i64 %66
%i648B

	full_text
	
i64 %66
8icmp8B.
,
	full_text

%68 = icmp sgt i32 %27, 255
%i328B

	full_text
	
i32 %27
=add8B4
2
	full_text%
#
!%69 = add i64 %65, -1099511627776
%i648B

	full_text
	
i64 %65
9ashr8B/
-
	full_text 

%70 = ashr exact i64 %69, 32
%i648B

	full_text
	
i64 %69
…getelementptr8Br
p
	full_textc
a
_%71 = getelementptr inbounds [256 x float], [256 x float]* @IMGVF_kernel.buffer, i64 0, i64 %70
%i648B

	full_text
	
i64 %70
<sitofp8B0
.
	full_text!

%72 = sitofp i32 %23 to float
%i328B

	full_text
	
i32 %23
5add8B,
*
	full_text

%73 = add nsw i32 %20, -1
%i328B

	full_text
	
i32 %20
5add8B,
*
	full_text

%74 = add nsw i32 %22, -1
%i328B

	full_text
	
i32 %22
?fsub8B5
3
	full_text&
$
"%75 = fsub float -0.000000e+00, %5
4fsub8B*
(
	full_text

%76 = fsub float %5, %6
4fadd8B*
(
	full_text

%77 = fadd float %5, %6
5fsub8B+
)
	full_text

%78 = fsub float %75, %6
)float8B

	full_text

	float %75
4fsub8B*
(
	full_text

%79 = fsub float %6, %5
@bitcast8B3
1
	full_text$
"
 %80 = bitcast float* %67 to i32*
+float*8B

	full_text


float* %67
5add8B,
*
	full_text

%81 = add nsw i32 %25, -1
%i328B

	full_text
	
i32 %25
8icmp8B.
,
	full_text

%82 = icmp slt i32 %27, 128
%i328B

	full_text
	
i32 %27
1shl8B(
&
	full_text

%83 = shl i64 %26, 32
%i648B

	full_text
	
i64 %26
;add8B2
0
	full_text#
!
%84 = add i64 %83, 549755813888
%i648B

	full_text
	
i64 %83
9ashr8B/
-
	full_text 

%85 = ashr exact i64 %84, 32
%i648B

	full_text
	
i64 %84
…getelementptr8Br
p
	full_textc
a
_%86 = getelementptr inbounds [256 x float], [256 x float]* @IMGVF_kernel.buffer, i64 0, i64 %85
%i648B

	full_text
	
i64 %85
7icmp8B-
+
	full_text

%87 = icmp slt i32 %27, 64
%i328B

	full_text
	
i32 %27
1shl8B(
&
	full_text

%88 = shl i64 %26, 32
%i648B

	full_text
	
i64 %26
;add8B2
0
	full_text#
!
%89 = add i64 %88, 274877906944
%i648B

	full_text
	
i64 %88
9ashr8B/
-
	full_text 

%90 = ashr exact i64 %89, 32
%i648B

	full_text
	
i64 %89
…getelementptr8Br
p
	full_textc
a
_%91 = getelementptr inbounds [256 x float], [256 x float]* @IMGVF_kernel.buffer, i64 0, i64 %90
%i648B

	full_text
	
i64 %90
7icmp8B-
+
	full_text

%92 = icmp slt i32 %27, 32
%i328B

	full_text
	
i32 %27
1shl8B(
&
	full_text

%93 = shl i64 %26, 32
%i648B

	full_text
	
i64 %26
;add8B2
0
	full_text#
!
%94 = add i64 %93, 137438953472
%i648B

	full_text
	
i64 %93
9ashr8B/
-
	full_text 

%95 = ashr exact i64 %94, 32
%i648B

	full_text
	
i64 %94
…getelementptr8Br
p
	full_textc
a
_%96 = getelementptr inbounds [256 x float], [256 x float]* @IMGVF_kernel.buffer, i64 0, i64 %95
%i648B

	full_text
	
i64 %95
7icmp8B-
+
	full_text

%97 = icmp slt i32 %27, 16
%i328B

	full_text
	
i32 %27
1shl8B(
&
	full_text

%98 = shl i64 %26, 32
%i648B

	full_text
	
i64 %26
:add8B1
/
	full_text"
 
%99 = add i64 %98, 68719476736
%i648B

	full_text
	
i64 %98
:ashr8B0
.
	full_text!

%100 = ashr exact i64 %99, 32
%i648B

	full_text
	
i64 %99
‡getelementptr8Bt
r
	full_texte
c
a%101 = getelementptr inbounds [256 x float], [256 x float]* @IMGVF_kernel.buffer, i64 0, i64 %100
&i648B

	full_text


i64 %100
7icmp8B-
+
	full_text

%102 = icmp slt i32 %27, 8
%i328B

	full_text
	
i32 %27
2shl8B)
'
	full_text

%103 = shl i64 %26, 32
%i648B

	full_text
	
i64 %26
<add8B3
1
	full_text$
"
 %104 = add i64 %103, 34359738368
&i648B

	full_text


i64 %103
;ashr8B1
/
	full_text"
 
%105 = ashr exact i64 %104, 32
&i648B

	full_text


i64 %104
‡getelementptr8Bt
r
	full_texte
c
a%106 = getelementptr inbounds [256 x float], [256 x float]* @IMGVF_kernel.buffer, i64 0, i64 %105
&i648B

	full_text


i64 %105
7icmp8B-
+
	full_text

%107 = icmp slt i32 %27, 4
%i328B

	full_text
	
i32 %27
2shl8B)
'
	full_text

%108 = shl i64 %26, 32
%i648B

	full_text
	
i64 %26
<add8B3
1
	full_text$
"
 %109 = add i64 %108, 17179869184
&i648B

	full_text


i64 %108
;ashr8B1
/
	full_text"
 
%110 = ashr exact i64 %109, 32
&i648B

	full_text


i64 %109
‡getelementptr8Bt
r
	full_texte
c
a%111 = getelementptr inbounds [256 x float], [256 x float]* @IMGVF_kernel.buffer, i64 0, i64 %110
&i648B

	full_text


i64 %110
7icmp8B-
+
	full_text

%112 = icmp slt i32 %27, 2
%i328B

	full_text
	
i32 %27
2shl8B)
'
	full_text

%113 = shl i64 %26, 32
%i648B

	full_text
	
i64 %26
;add8B2
0
	full_text#
!
%114 = add i64 %113, 8589934592
&i648B

	full_text


i64 %113
;ashr8B1
/
	full_text"
 
%115 = ashr exact i64 %114, 32
&i648B

	full_text


i64 %114
‡getelementptr8Bt
r
	full_texte
c
a%116 = getelementptr inbounds [256 x float], [256 x float]* @IMGVF_kernel.buffer, i64 0, i64 %115
&i648B

	full_text


i64 %115
7icmp8B-
+
	full_text

%117 = icmp slt i32 %27, 1
%i328B

	full_text
	
i32 %27
2shl8B)
'
	full_text

%118 = shl i64 %26, 32
%i648B

	full_text
	
i64 %26
;add8B2
0
	full_text#
!
%119 = add i64 %118, 4294967296
&i648B

	full_text


i64 %118
;ashr8B1
/
	full_text"
 
%120 = ashr exact i64 %119, 32
&i648B

	full_text


i64 %119
‡getelementptr8Bt
r
	full_texte
c
a%121 = getelementptr inbounds [256 x float], [256 x float]* @IMGVF_kernel.buffer, i64 0, i64 %120
&i648B

	full_text


i64 %120
(br8B 

	full_text

br label %122
Gphi8	B>
<
	full_text/
-
+%123 = phi i32 [ %50, %62 ], [ %272, %288 ]
%i328	B

	full_text
	
i32 %50
&i328	B

	full_text


i32 %272
Ephi8	B<
:
	full_text-
+
)%124 = phi i32 [ 0, %62 ], [ %289, %288 ]
&i328	B

	full_text


i32 %289
<br8	B4
2
	full_text%
#
!br i1 %28, label %125, label %270
#i18	B

	full_text


i1 %28
(br8
B 

	full_text

br label %126
Fphi8B=
;
	full_text.
,
*%127 = phi i32 [ %268, %264 ], [ 0, %125 ]
&i328B

	full_text


i32 %268
Iphi8B@
>
	full_text1
/
-%128 = phi i32 [ %135, %264 ], [ %123, %125 ]
&i328B

	full_text


i32 %135
&i328B

	full_text


i32 %123
Hphi8B?
=
	full_text0
.
,%129 = phi i32 [ %139, %264 ], [ %64, %125 ]
&i328B

	full_text


i32 %139
%i328B

	full_text
	
i32 %64
Sphi8BJ
H
	full_text;
9
7%130 = phi float [ %267, %264 ], [ 0.000000e+00, %125 ]
*float8B

	full_text


float %267
2shl8B)
'
	full_text

%131 = shl i32 %127, 8
&i328B

	full_text


i32 %127
8add8B/
-
	full_text 

%132 = add nsw i32 %131, %27
&i328B

	full_text


i32 %131
%i328B

	full_text
	
i32 %27
>sitofp8B2
0
	full_text#
!
%133 = sitofp i32 %132 to float
&i328B

	full_text


i32 %132
8fmul8B.
,
	full_text

%134 = fmul float %55, %133
)float8B

	full_text

	float %55
*float8B

	full_text


float %133
>fptosi8B2
0
	full_text#
!
%135 = fptosi float %134 to i32
*float8B

	full_text


float %134
8add8B/
-
	full_text 

%136 = add nsw i32 %129, %56
&i328B

	full_text


i32 %129
%i328B

	full_text
	
i32 %56
:icmp8B0
.
	full_text!

%137 = icmp slt i32 %136, %22
&i328B

	full_text


i32 %136
%i328B

	full_text
	
i32 %22
Dselect8B8
6
	full_text)
'
%%138 = select i1 %137, i32 0, i32 %22
$i18B

	full_text
	
i1 %137
%i328B

	full_text
	
i32 %22
9sub8B0
.
	full_text!

%139 = sub nsw i32 %136, %138
&i328B

	full_text


i32 %136
&i328B

	full_text


i32 %138
:icmp8B0
.
	full_text!

%140 = icmp sgt i32 %20, %135
%i328B

	full_text
	
i32 %20
&i328B

	full_text


i32 %135
=br8B5
3
	full_text&
$
"br i1 %140, label %141, label %241
$i18B

	full_text
	
i1 %140
7icmp8B-
+
	full_text

%142 = icmp eq i32 %135, 0
&i328B

	full_text


i32 %135
7add8B.
,
	full_text

%143 = add nsw i32 %135, -1
&i328B

	full_text


i32 %135
Eselect8B9
7
	full_text*
(
&%144 = select i1 %142, i32 0, i32 %143
$i18B

	full_text
	
i1 %142
&i328B

	full_text


i32 %143
9icmp8B/
-
	full_text 

%145 = icmp eq i32 %73, %135
%i328B

	full_text
	
i32 %73
&i328B

	full_text


i32 %135
6add8B-
+
	full_text

%146 = add nsw i32 %135, 1
&i328B

	full_text


i32 %135
Gselect8B;
9
	full_text,
*
(%147 = select i1 %145, i32 %73, i32 %146
$i18B

	full_text
	
i1 %145
%i328B

	full_text
	
i32 %73
&i328B

	full_text


i32 %146
7icmp8B-
+
	full_text

%148 = icmp eq i32 %139, 0
&i328B

	full_text


i32 %139
7add8B.
,
	full_text

%149 = add nsw i32 %139, -1
&i328B

	full_text


i32 %139
Eselect8B9
7
	full_text*
(
&%150 = select i1 %148, i32 0, i32 %149
$i18B

	full_text
	
i1 %148
&i328B

	full_text


i32 %149
9icmp8B/
-
	full_text 

%151 = icmp eq i32 %139, %74
&i328B

	full_text


i32 %139
%i328B

	full_text
	
i32 %74
6add8B-
+
	full_text

%152 = add nsw i32 %139, 1
&i328B

	full_text


i32 %139
Gselect8B;
9
	full_text,
*
(%153 = select i1 %151, i32 %74, i32 %152
$i18B

	full_text
	
i1 %151
%i328B

	full_text
	
i32 %74
&i328B

	full_text


i32 %152
8mul8B/
-
	full_text 

%154 = mul nsw i32 %22, %135
%i328B

	full_text
	
i32 %22
&i328B

	full_text


i32 %135
9add8B0
.
	full_text!

%155 = add nsw i32 %154, %139
&i328B

	full_text


i32 %154
&i328B

	full_text


i32 %139
8sext8B.
,
	full_text

%156 = sext i32 %155 to i64
&i328B

	full_text


i32 %155
ˆgetelementptr8Bu
s
	full_textf
d
b%157 = getelementptr inbounds [3321 x float], [3321 x float]* @IMGVF_kernel.IMGVF, i64 0, i64 %156
&i648B

	full_text


i64 %156
Oload8BE
C
	full_text6
4
2%158 = load float, float* %157, align 4, !tbaa !12
,float*8B

	full_text

float* %157
8mul8B/
-
	full_text 

%159 = mul nsw i32 %144, %22
&i328B

	full_text


i32 %144
%i328B

	full_text
	
i32 %22
9add8B0
.
	full_text!

%160 = add nsw i32 %159, %139
&i328B

	full_text


i32 %159
&i328B

	full_text


i32 %139
8sext8B.
,
	full_text

%161 = sext i32 %160 to i64
&i328B

	full_text


i32 %160
ˆgetelementptr8Bu
s
	full_textf
d
b%162 = getelementptr inbounds [3321 x float], [3321 x float]* @IMGVF_kernel.IMGVF, i64 0, i64 %161
&i648B

	full_text


i64 %161
Oload8BE
C
	full_text6
4
2%163 = load float, float* %162, align 4, !tbaa !12
,float*8B

	full_text

float* %162
9fsub8B/
-
	full_text 

%164 = fsub float %163, %158
*float8B

	full_text


float %163
*float8B

	full_text


float %158
8mul8B/
-
	full_text 

%165 = mul nsw i32 %147, %22
&i328B

	full_text


i32 %147
%i328B

	full_text
	
i32 %22
9add8B0
.
	full_text!

%166 = add nsw i32 %165, %139
&i328B

	full_text


i32 %165
&i328B

	full_text


i32 %139
8sext8B.
,
	full_text

%167 = sext i32 %166 to i64
&i328B

	full_text


i32 %166
ˆgetelementptr8Bu
s
	full_textf
d
b%168 = getelementptr inbounds [3321 x float], [3321 x float]* @IMGVF_kernel.IMGVF, i64 0, i64 %167
&i648B

	full_text


i64 %167
Oload8BE
C
	full_text6
4
2%169 = load float, float* %168, align 4, !tbaa !12
,float*8B

	full_text

float* %168
9fsub8B/
-
	full_text 

%170 = fsub float %169, %158
*float8B

	full_text


float %169
*float8B

	full_text


float %158
9add8B0
.
	full_text!

%171 = add nsw i32 %154, %150
&i328B

	full_text


i32 %154
&i328B

	full_text


i32 %150
8sext8B.
,
	full_text

%172 = sext i32 %171 to i64
&i328B

	full_text


i32 %171
ˆgetelementptr8Bu
s
	full_textf
d
b%173 = getelementptr inbounds [3321 x float], [3321 x float]* @IMGVF_kernel.IMGVF, i64 0, i64 %172
&i648B

	full_text


i64 %172
Oload8BE
C
	full_text6
4
2%174 = load float, float* %173, align 4, !tbaa !12
,float*8B

	full_text

float* %173
9fsub8B/
-
	full_text 

%175 = fsub float %174, %158
*float8B

	full_text


float %174
*float8B

	full_text


float %158
9add8B0
.
	full_text!

%176 = add nsw i32 %154, %153
&i328B

	full_text


i32 %154
&i328B

	full_text


i32 %153
8sext8B.
,
	full_text

%177 = sext i32 %176 to i64
&i328B

	full_text


i32 %176
ˆgetelementptr8Bu
s
	full_textf
d
b%178 = getelementptr inbounds [3321 x float], [3321 x float]* @IMGVF_kernel.IMGVF, i64 0, i64 %177
&i648B

	full_text


i64 %177
Oload8BE
C
	full_text6
4
2%179 = load float, float* %178, align 4, !tbaa !12
,float*8B

	full_text

float* %178
9fsub8B/
-
	full_text 

%180 = fsub float %179, %158
*float8B

	full_text


float %179
*float8B

	full_text


float %158
9add8B0
.
	full_text!

%181 = add nsw i32 %159, %153
&i328B

	full_text


i32 %159
&i328B

	full_text


i32 %153
8sext8B.
,
	full_text

%182 = sext i32 %181 to i64
&i328B

	full_text


i32 %181
ˆgetelementptr8Bu
s
	full_textf
d
b%183 = getelementptr inbounds [3321 x float], [3321 x float]* @IMGVF_kernel.IMGVF, i64 0, i64 %182
&i648B

	full_text


i64 %182
Oload8BE
C
	full_text6
4
2%184 = load float, float* %183, align 4, !tbaa !12
,float*8B

	full_text

float* %183
9fsub8B/
-
	full_text 

%185 = fsub float %184, %158
*float8B

	full_text


float %184
*float8B

	full_text


float %158
9add8B0
.
	full_text!

%186 = add nsw i32 %165, %153
&i328B

	full_text


i32 %165
&i328B

	full_text


i32 %153
8sext8B.
,
	full_text

%187 = sext i32 %186 to i64
&i328B

	full_text


i32 %186
ˆgetelementptr8Bu
s
	full_textf
d
b%188 = getelementptr inbounds [3321 x float], [3321 x float]* @IMGVF_kernel.IMGVF, i64 0, i64 %187
&i648B

	full_text


i64 %187
Oload8BE
C
	full_text6
4
2%189 = load float, float* %188, align 4, !tbaa !12
,float*8B

	full_text

float* %188
9fsub8B/
-
	full_text 

%190 = fsub float %189, %158
*float8B

	full_text


float %189
*float8B

	full_text


float %158
9add8B0
.
	full_text!

%191 = add nsw i32 %159, %150
&i328B

	full_text


i32 %159
&i328B

	full_text


i32 %150
8sext8B.
,
	full_text

%192 = sext i32 %191 to i64
&i328B

	full_text


i32 %191
ˆgetelementptr8Bu
s
	full_textf
d
b%193 = getelementptr inbounds [3321 x float], [3321 x float]* @IMGVF_kernel.IMGVF, i64 0, i64 %192
&i648B

	full_text


i64 %192
Oload8BE
C
	full_text6
4
2%194 = load float, float* %193, align 4, !tbaa !12
,float*8B

	full_text

float* %193
9fsub8B/
-
	full_text 

%195 = fsub float %194, %158
*float8B

	full_text


float %194
*float8B

	full_text


float %158
9add8B0
.
	full_text!

%196 = add nsw i32 %165, %150
&i328B

	full_text


i32 %165
&i328B

	full_text


i32 %150
8sext8B.
,
	full_text

%197 = sext i32 %196 to i64
&i328B

	full_text


i32 %196
ˆgetelementptr8Bu
s
	full_textf
d
b%198 = getelementptr inbounds [3321 x float], [3321 x float]* @IMGVF_kernel.IMGVF, i64 0, i64 %197
&i648B

	full_text


i64 %197
Oload8BE
C
	full_text6
4
2%199 = load float, float* %198, align 4, !tbaa !12
,float*8B

	full_text

float* %198
9fsub8B/
-
	full_text 

%200 = fsub float %199, %158
*float8B

	full_text


float %199
*float8B

	full_text


float %158
7fmul8B-
+
	full_text

%201 = fmul float %164, %6
*float8B

	full_text


float %164
8fmul8B.
,
	full_text

%202 = fmul float %57, %201
)float8B

	full_text

	float %57
*float8B

	full_text


float %201
Bfsub8B8
6
	full_text)
'
%%203 = fsub float -0.000000e+00, %202
*float8B

	full_text


float %202
Mcall8BC
A
	full_text4
2
0%204 = tail call float @heaviside(float %203) #7
*float8B

	full_text


float %203
7fmul8B-
+
	full_text

%205 = fmul float %170, %6
*float8B

	full_text


float %170
8fmul8B.
,
	full_text

%206 = fmul float %57, %205
)float8B

	full_text

	float %57
*float8B

	full_text


float %205
Mcall8BC
A
	full_text4
2
0%207 = tail call float @heaviside(float %206) #7
*float8B

	full_text


float %206
8fmul8B.
,
	full_text

%208 = fmul float %175, %75
*float8B

	full_text


float %175
)float8B

	full_text

	float %75
8fmul8B.
,
	full_text

%209 = fmul float %57, %208
)float8B

	full_text

	float %57
*float8B

	full_text


float %208
Mcall8BC
A
	full_text4
2
0%210 = tail call float @heaviside(float %209) #7
*float8B

	full_text


float %209
7fmul8B-
+
	full_text

%211 = fmul float %180, %5
*float8B

	full_text


float %180
8fmul8B.
,
	full_text

%212 = fmul float %57, %211
)float8B

	full_text

	float %57
*float8B

	full_text


float %211
Mcall8BC
A
	full_text4
2
0%213 = tail call float @heaviside(float %212) #7
*float8B

	full_text


float %212
8fmul8B.
,
	full_text

%214 = fmul float %76, %185
)float8B

	full_text

	float %76
*float8B

	full_text


float %185
8fmul8B.
,
	full_text

%215 = fmul float %57, %214
)float8B

	full_text

	float %57
*float8B

	full_text


float %214
Mcall8BC
A
	full_text4
2
0%216 = tail call float @heaviside(float %215) #7
*float8B

	full_text


float %215
8fmul8B.
,
	full_text

%217 = fmul float %77, %190
)float8B

	full_text

	float %77
*float8B

	full_text


float %190
8fmul8B.
,
	full_text

%218 = fmul float %57, %217
)float8B

	full_text

	float %57
*float8B

	full_text


float %217
Mcall8BC
A
	full_text4
2
0%219 = tail call float @heaviside(float %218) #7
*float8B

	full_text


float %218
8fmul8B.
,
	full_text

%220 = fmul float %78, %195
)float8B

	full_text

	float %78
*float8B

	full_text


float %195
8fmul8B.
,
	full_text

%221 = fmul float %57, %220
)float8B

	full_text

	float %57
*float8B

	full_text


float %220
Mcall8BC
A
	full_text4
2
0%222 = tail call float @heaviside(float %221) #7
*float8B

	full_text


float %221
8fmul8B.
,
	full_text

%223 = fmul float %79, %200
)float8B

	full_text

	float %79
*float8B

	full_text


float %200
8fmul8B.
,
	full_text

%224 = fmul float %57, %223
)float8B

	full_text

	float %57
*float8B

	full_text


float %223
Mcall8BC
A
	full_text4
2
0%225 = tail call float @heaviside(float %224) #7
*float8B

	full_text


float %224
9fmul8B/
-
	full_text 

%226 = fmul float %170, %207
*float8B

	full_text


float %170
*float8B

	full_text


float %207
icall8B_
]
	full_textP
N
L%227 = tail call float @llvm.fmuladd.f32(float %204, float %164, float %226)
*float8B

	full_text


float %204
*float8B

	full_text


float %164
*float8B

	full_text


float %226
icall8B_
]
	full_textP
N
L%228 = tail call float @llvm.fmuladd.f32(float %210, float %175, float %227)
*float8B

	full_text


float %210
*float8B

	full_text


float %175
*float8B

	full_text


float %227
icall8B_
]
	full_textP
N
L%229 = tail call float @llvm.fmuladd.f32(float %213, float %180, float %228)
*float8B

	full_text


float %213
*float8B

	full_text


float %180
*float8B

	full_text


float %228
icall8B_
]
	full_textP
N
L%230 = tail call float @llvm.fmuladd.f32(float %216, float %185, float %229)
*float8B

	full_text


float %216
*float8B

	full_text


float %185
*float8B

	full_text


float %229
icall8B_
]
	full_textP
N
L%231 = tail call float @llvm.fmuladd.f32(float %219, float %190, float %230)
*float8B

	full_text


float %219
*float8B

	full_text


float %190
*float8B

	full_text


float %230
icall8B_
]
	full_textP
N
L%232 = tail call float @llvm.fmuladd.f32(float %222, float %195, float %231)
*float8B

	full_text


float %222
*float8B

	full_text


float %195
*float8B

	full_text


float %231
icall8B_
]
	full_textP
N
L%233 = tail call float @llvm.fmuladd.f32(float %225, float %200, float %232)
*float8B

	full_text


float %225
*float8B

	full_text


float %200
*float8B

	full_text


float %232
wcall8Bm
k
	full_text^
\
Z%234 = tail call float @llvm.fmuladd.f32(float %233, float 0x3FB99999A0000000, float %158)
*float8B

	full_text


float %233
*float8B

	full_text


float %158
_getelementptr8BL
J
	full_text=
;
9%235 = getelementptr inbounds float, float* %18, i64 %156
+float*8B

	full_text


float* %18
&i648B

	full_text


i64 %156
Oload8BE
C
	full_text6
4
2%236 = load float, float* %235, align 4, !tbaa !12
,float*8B

	full_text

float* %235
Gfmul8B=
;
	full_text.
,
*%237 = fmul float %236, 0x3FC99999A0000000
*float8B

	full_text


float %236
9fsub8B/
-
	full_text 

%238 = fsub float %234, %236
*float8B

	full_text


float %234
*float8B

	full_text


float %236
Bfsub8B8
6
	full_text)
'
%%239 = fsub float -0.000000e+00, %237
*float8B

	full_text


float %237
icall8B_
]
	full_textP
N
L%240 = tail call float @llvm.fmuladd.f32(float %239, float %238, float %234)
*float8B

	full_text


float %239
*float8B

	full_text


float %238
*float8B

	full_text


float %234
(br8B 

	full_text

br label %241
Sphi8BJ
H
	full_text;
9
7%242 = phi float [ %158, %141 ], [ 0.000000e+00, %126 ]
*float8B

	full_text


float %158
Sphi8BJ
H
	full_text;
9
7%243 = phi float [ %240, %141 ], [ 0.000000e+00, %126 ]
*float8B

	full_text


float %240
7icmp8B-
+
	full_text

%244 = icmp ne i32 %127, 0
&i328B

	full_text


i32 %127
:icmp8B0
.
	full_text!

%245 = icmp slt i32 %128, %20
&i328B

	full_text


i32 %128
%i328B

	full_text
	
i32 %20
4and8B+
)
	full_text

%246 = and i1 %245, %244
$i18B

	full_text
	
i1 %245
$i18B

	full_text
	
i1 %244
=br8B5
3
	full_text&
$
"br i1 %246, label %247, label %254
$i18B

	full_text
	
i1 %246
Jload8B@
>
	full_text1
/
-%248 = load i32, i32* %80, align 4, !tbaa !12
'i32*8B

	full_text


i32* %80
8mul8B/
-
	full_text 

%249 = mul nsw i32 %128, %22
&i328B

	full_text


i32 %128
%i328B

	full_text
	
i32 %22
9add8B0
.
	full_text!

%250 = add nsw i32 %249, %129
&i328B

	full_text


i32 %249
&i328B

	full_text


i32 %129
8sext8B.
,
	full_text

%251 = sext i32 %250 to i64
&i328B

	full_text


i32 %250
ˆgetelementptr8Bu
s
	full_textf
d
b%252 = getelementptr inbounds [3321 x float], [3321 x float]* @IMGVF_kernel.IMGVF, i64 0, i64 %251
&i648B

	full_text


i64 %251
Bbitcast8B5
3
	full_text&
$
"%253 = bitcast float* %252 to i32*
,float*8B

	full_text

float* %252
Kstore8B@
>
	full_text1
/
-store i32 %248, i32* %253, align 4, !tbaa !12
&i328B

	full_text


i32 %248
(i32*8B

	full_text

	i32* %253
(br8B 

	full_text

br label %254
:icmp8B0
.
	full_text!

%255 = icmp slt i32 %127, %81
&i328B

	full_text


i32 %127
%i328B

	full_text
	
i32 %81
=br8B5
3
	full_text&
$
"br i1 %255, label %262, label %256
$i18B

	full_text
	
i1 %255
=br8B5
3
	full_text&
$
"br i1 %140, label %257, label %264
$i18B

	full_text
	
i1 %140
8mul8B/
-
	full_text 

%258 = mul nsw i32 %22, %135
%i328B

	full_text
	
i32 %22
&i328B

	full_text


i32 %135
9add8B0
.
	full_text!

%259 = add nsw i32 %258, %139
&i328B

	full_text


i32 %258
&i328B

	full_text


i32 %139
8sext8B.
,
	full_text

%260 = sext i32 %259 to i64
&i328B

	full_text


i32 %259
ˆgetelementptr8Bu
s
	full_textf
d
b%261 = getelementptr inbounds [3321 x float], [3321 x float]* @IMGVF_kernel.IMGVF, i64 0, i64 %260
&i648B

	full_text


i64 %260
(br8B 

	full_text

br label %262
Kphi8BB
@
	full_text3
1
/%263 = phi float* [ %261, %257 ], [ %67, %254 ]
,float*8B

	full_text

float* %261
+float*8B

	full_text


float* %67
Ostore8BD
B
	full_text5
3
1store float %243, float* %263, align 4, !tbaa !12
*float8B

	full_text


float %243
,float*8B

	full_text

float* %263
(br8B 

	full_text

br label %264
9fsub8B/
-
	full_text 

%265 = fsub float %243, %242
*float8B

	full_text


float %243
*float8B

	full_text


float %242
Lcall8BB
@
	full_text3
1
/%266 = tail call float @_Z4fabsf(float %265) #5
*float8B

	full_text


float %265
9fadd8B/
-
	full_text 

%267 = fadd float %130, %266
*float8B

	full_text


float %130
*float8B

	full_text


float %266
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #6
:add8B1
/
	full_text"
 
%268 = add nuw nsw i32 %127, 1
&i328B

	full_text


i32 %127
:icmp8B0
.
	full_text!

%269 = icmp slt i32 %268, %25
&i328B

	full_text


i32 %268
%i328B

	full_text
	
i32 %25
=br8B5
3
	full_text&
$
"br i1 %269, label %126, label %270
$i18B

	full_text
	
i1 %269
Sphi8BJ
H
	full_text;
9
7%271 = phi float [ 0.000000e+00, %122 ], [ %267, %264 ]
*float8B

	full_text


float %267
Iphi8B@
>
	full_text1
/
-%272 = phi i32 [ %123, %122 ], [ %135, %264 ]
&i328B

	full_text


i32 %123
&i328B

	full_text


i32 %135
Nstore8BC
A
	full_text4
2
0store float %271, float* %67, align 4, !tbaa !12
*float8B

	full_text


float %271
+float*8B

	full_text


float* %67
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #6
<br8B4
2
	full_text%
#
!br i1 %68, label %273, label %277
#i18B

	full_text


i1 %68
Nload8BD
B
	full_text5
3
1%274 = load float, float* %67, align 4, !tbaa !12
+float*8B

	full_text


float* %67
Nload8BD
B
	full_text5
3
1%275 = load float, float* %71, align 4, !tbaa !12
+float*8B

	full_text


float* %71
9fadd8B/
-
	full_text 

%276 = fadd float %274, %275
*float8B

	full_text


float %274
*float8B

	full_text


float %275
Nstore8BC
A
	full_text4
2
0store float %276, float* %71, align 4, !tbaa !12
*float8B

	full_text


float %276
+float*8B

	full_text


float* %71
(br8B 

	full_text

br label %277
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #6
<br8B4
2
	full_text%
#
!br i1 %82, label %278, label %282
#i18B

	full_text


i1 %82
Nload8BD
B
	full_text5
3
1%279 = load float, float* %86, align 4, !tbaa !12
+float*8B

	full_text


float* %86
Nload8BD
B
	full_text5
3
1%280 = load float, float* %67, align 4, !tbaa !12
+float*8B

	full_text


float* %67
9fadd8B/
-
	full_text 

%281 = fadd float %279, %280
*float8B

	full_text


float %279
*float8B

	full_text


float %280
Nstore8BC
A
	full_text4
2
0store float %281, float* %67, align 4, !tbaa !12
*float8B

	full_text


float %281
+float*8B

	full_text


float* %67
(br8B 

	full_text

br label %282
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #6
<br8B4
2
	full_text%
#
!br i1 %87, label %316, label %320
#i18B

	full_text


i1 %87
Nload8BD
B
	full_text5
3
1%284 = load float, float* %67, align 4, !tbaa !12
+float*8B

	full_text


float* %67
Efdiv8B;
9
	full_text,
*
(%285 = fdiv float %284, %72, !fpmath !14
*float8B

	full_text


float %284
)float8B

	full_text

	float %72
;fcmp8B1
/
	full_text"
 
%286 = fcmp olt float %285, %9
*float8B

	full_text


float %285
=br8B5
3
	full_text&
$
"br i1 %286, label %287, label %288
$i18B

	full_text
	
i1 %286
_store8BT
R
	full_textE
C
Astore i32 1, i32* @IMGVF_kernel.cell_converged, align 4, !tbaa !8
(br8B 

	full_text

br label %288
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #6
:add8B1
/
	full_text"
 
%289 = add nuw nsw i32 %124, 1
&i328B

	full_text


i32 %124
bload8BX
V
	full_textI
G
E%290 = load i32, i32* @IMGVF_kernel.cell_converged, align 4, !tbaa !8
7icmp8B-
+
	full_text

%291 = icmp eq i32 %290, 0
&i328B

	full_text


i32 %290
9icmp8B/
-
	full_text 

%292 = icmp slt i32 %289, %8
&i328B

	full_text


i32 %289
4and8B+
)
	full_text

%293 = and i1 %292, %291
$i18B

	full_text
	
i1 %292
$i18B

	full_text
	
i1 %291
=br8B5
3
	full_text&
$
"br i1 %293, label %122, label %294
$i18B

	full_text
	
i1 %293
<br8B4
2
	full_text%
#
!br i1 %28, label %295, label %315
#i18B

	full_text


i1 %28
(br8B 

	full_text

br label %296
Fphi8B=
;
	full_text.
,
*%297 = phi i32 [ %313, %312 ], [ 0, %295 ]
&i328B

	full_text


i32 %313
6shl8B-
+
	full_text

%298 = shl nsw i32 %297, 8
&i328B

	full_text


i32 %297
8add8B/
-
	full_text 

%299 = add nsw i32 %298, %27
&i328B

	full_text


i32 %298
%i328B

	full_text
	
i32 %27
6sdiv8B,
*
	full_text

%300 = sdiv i32 %299, %22
&i328B

	full_text


i32 %299
%i328B

	full_text
	
i32 %22
:icmp8B0
.
	full_text!

%301 = icmp slt i32 %300, %20
&i328B

	full_text


i32 %300
%i328B

	full_text
	
i32 %20
=br8B5
3
	full_text&
$
"br i1 %301, label %302, label %312
$i18B

	full_text
	
i1 %301
6srem8B,
*
	full_text

%303 = srem i32 %299, %22
&i328B

	full_text


i32 %299
%i328B

	full_text
	
i32 %22
8mul8B/
-
	full_text 

%304 = mul nsw i32 %300, %22
&i328B

	full_text


i32 %300
%i328B

	full_text
	
i32 %22
9add8B0
.
	full_text!

%305 = add nsw i32 %303, %304
&i328B

	full_text


i32 %303
&i328B

	full_text


i32 %304
8sext8B.
,
	full_text

%306 = sext i32 %305 to i64
&i328B

	full_text


i32 %305
ˆgetelementptr8Bu
s
	full_textf
d
b%307 = getelementptr inbounds [3321 x float], [3321 x float]* @IMGVF_kernel.IMGVF, i64 0, i64 %306
&i648B

	full_text


i64 %306
Bbitcast8B5
3
	full_text&
$
"%308 = bitcast float* %307 to i32*
,float*8B

	full_text

float* %307
Kload8BA
?
	full_text2
0
.%309 = load i32, i32* %308, align 4, !tbaa !12
(i32*8B

	full_text

	i32* %308
_getelementptr8BL
J
	full_text=
;
9%310 = getelementptr inbounds float, float* %17, i64 %306
+float*8B

	full_text


float* %17
&i648B

	full_text


i64 %306
Bbitcast8B5
3
	full_text&
$
"%311 = bitcast float* %310 to i32*
,float*8B

	full_text

float* %310
Kstore8B@
>
	full_text1
/
-store i32 %309, i32* %311, align 4, !tbaa !12
&i328B

	full_text


i32 %309
(i32*8B

	full_text

	i32* %311
(br8B 

	full_text

br label %312
:add8 B1
/
	full_text"
 
%313 = add nuw nsw i32 %297, 1
&i328 B

	full_text


i32 %297
:icmp8 B0
.
	full_text!

%314 = icmp slt i32 %313, %25
&i328 B

	full_text


i32 %313
%i328 B

	full_text
	
i32 %25
=br8 B5
3
	full_text&
$
"br i1 %314, label %296, label %315
$i18 B

	full_text
	
i1 %314
$ret8!B

	full_text


ret void
Nload8"BD
B
	full_text5
3
1%317 = load float, float* %91, align 4, !tbaa !12
+float*8"B

	full_text


float* %91
Nload8"BD
B
	full_text5
3
1%318 = load float, float* %67, align 4, !tbaa !12
+float*8"B

	full_text


float* %67
9fadd8"B/
-
	full_text 

%319 = fadd float %317, %318
*float8"B

	full_text


float %317
*float8"B

	full_text


float %318
Nstore8"BC
A
	full_text4
2
0store float %319, float* %67, align 4, !tbaa !12
*float8"B

	full_text


float %319
+float*8"B

	full_text


float* %67
(br8"B 

	full_text

br label %320
Bcall8#B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #6
<br8#B4
2
	full_text%
#
!br i1 %92, label %321, label %325
#i18#B

	full_text


i1 %92
Nload8$BD
B
	full_text5
3
1%322 = load float, float* %96, align 4, !tbaa !12
+float*8$B

	full_text


float* %96
Nload8$BD
B
	full_text5
3
1%323 = load float, float* %67, align 4, !tbaa !12
+float*8$B

	full_text


float* %67
9fadd8$B/
-
	full_text 

%324 = fadd float %322, %323
*float8$B

	full_text


float %322
*float8$B

	full_text


float %323
Nstore8$BC
A
	full_text4
2
0store float %324, float* %67, align 4, !tbaa !12
*float8$B

	full_text


float %324
+float*8$B

	full_text


float* %67
(br8$B 

	full_text

br label %325
Bcall8%B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #6
<br8%B4
2
	full_text%
#
!br i1 %97, label %326, label %330
#i18%B

	full_text


i1 %97
Oload8&BE
C
	full_text6
4
2%327 = load float, float* %101, align 4, !tbaa !12
,float*8&B

	full_text

float* %101
Nload8&BD
B
	full_text5
3
1%328 = load float, float* %67, align 4, !tbaa !12
+float*8&B

	full_text


float* %67
9fadd8&B/
-
	full_text 

%329 = fadd float %327, %328
*float8&B

	full_text


float %327
*float8&B

	full_text


float %328
Nstore8&BC
A
	full_text4
2
0store float %329, float* %67, align 4, !tbaa !12
*float8&B

	full_text


float %329
+float*8&B

	full_text


float* %67
(br8&B 

	full_text

br label %330
Bcall8'B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #6
=br8'B5
3
	full_text&
$
"br i1 %102, label %331, label %335
$i18'B

	full_text
	
i1 %102
Oload8(BE
C
	full_text6
4
2%332 = load float, float* %106, align 4, !tbaa !12
,float*8(B

	full_text

float* %106
Nload8(BD
B
	full_text5
3
1%333 = load float, float* %67, align 4, !tbaa !12
+float*8(B

	full_text


float* %67
9fadd8(B/
-
	full_text 

%334 = fadd float %332, %333
*float8(B

	full_text


float %332
*float8(B

	full_text


float %333
Nstore8(BC
A
	full_text4
2
0store float %334, float* %67, align 4, !tbaa !12
*float8(B

	full_text


float %334
+float*8(B

	full_text


float* %67
(br8(B 

	full_text

br label %335
Bcall8)B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #6
=br8)B5
3
	full_text&
$
"br i1 %107, label %336, label %340
$i18)B

	full_text
	
i1 %107
Oload8*BE
C
	full_text6
4
2%337 = load float, float* %111, align 4, !tbaa !12
,float*8*B

	full_text

float* %111
Nload8*BD
B
	full_text5
3
1%338 = load float, float* %67, align 4, !tbaa !12
+float*8*B

	full_text


float* %67
9fadd8*B/
-
	full_text 

%339 = fadd float %337, %338
*float8*B

	full_text


float %337
*float8*B

	full_text


float %338
Nstore8*BC
A
	full_text4
2
0store float %339, float* %67, align 4, !tbaa !12
*float8*B

	full_text


float %339
+float*8*B

	full_text


float* %67
(br8*B 

	full_text

br label %340
Bcall8+B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #6
=br8+B5
3
	full_text&
$
"br i1 %112, label %341, label %345
$i18+B

	full_text
	
i1 %112
Oload8,BE
C
	full_text6
4
2%342 = load float, float* %116, align 4, !tbaa !12
,float*8,B

	full_text

float* %116
Nload8,BD
B
	full_text5
3
1%343 = load float, float* %67, align 4, !tbaa !12
+float*8,B

	full_text


float* %67
9fadd8,B/
-
	full_text 

%344 = fadd float %342, %343
*float8,B

	full_text


float %342
*float8,B

	full_text


float %343
Nstore8,BC
A
	full_text4
2
0store float %344, float* %67, align 4, !tbaa !12
*float8,B

	full_text


float %344
+float*8,B

	full_text


float* %67
(br8,B 

	full_text

br label %345
Bcall8-B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #6
=br8-B5
3
	full_text&
$
"br i1 %117, label %346, label %350
$i18-B

	full_text
	
i1 %117
Oload8.BE
C
	full_text6
4
2%347 = load float, float* %121, align 4, !tbaa !12
,float*8.B

	full_text

float* %121
Nload8.BD
B
	full_text5
3
1%348 = load float, float* %67, align 4, !tbaa !12
+float*8.B

	full_text


float* %67
9fadd8.B/
-
	full_text 

%349 = fadd float %347, %348
*float8.B

	full_text


float %347
*float8.B

	full_text


float %348
Nstore8.BC
A
	full_text4
2
0store float %349, float* %67, align 4, !tbaa !12
*float8.B

	full_text


float %349
+float*8.B

	full_text


float* %67
(br8.B 

	full_text

br label %350
Bcall8/B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #6
<br8/B4
2
	full_text%
#
!br i1 %51, label %283, label %288
#i18/B

	full_text


i1 %51
*float*80B

	full_text

	float* %1
(float80B

	full_text


float %9
$i3280B

	full_text


i32 %8
&i32*80B

	full_text
	
i32* %4
*float*80B

	full_text

	float* %0
(float80B

	full_text


float %5
&i32*80B

	full_text
	
i32* %3
(float80B

	full_text


float %7
&i32*80B

	full_text
	
i32* %2
(float80B

	full_text


float %6
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
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
,i6480B!

	full_text

i64 8589934592
8float80B+
)
	full_text

float 0x3FC99999A0000000
#i3280B

	full_text	

i32 4
-i6480B"
 
	full_text

i64 17179869184
%i3280B

	full_text
	
i32 255
%i3280B

	full_text
	
i32 256
#i3280B

	full_text	

i32 2
#i6480B

	full_text	

i64 0
#i3280B

	full_text	

i32 1
-i6480B"
 
	full_text

i64 34359738368
,i6480B!

	full_text

i64 4294967296
$i3280B

	full_text


i32 16
mi32*80Ba
_
	full_textR
P
N@IMGVF_kernel.cell_converged = internal unnamed_addr global i32 undef, align 4
#i3280B

	full_text	

i32 8
.i6480B#
!
	full_text

i64 137438953472
$i3280B

	full_text


i32 64
-i6480B"
 
	full_text

i64 68719476736
2float80B%
#
	full_text

float 0.000000e+00
8float80B+
)
	full_text

float 0x3FB99999A0000000
.i6480B#
!
	full_text

i64 274877906944
.i6480B#
!
	full_text

i64 549755813888
3float80B&
$
	full_text

float -0.000000e+00
$i3280B

	full_text


i32 -1
z[256 x float]*80Bd
b
	full_textU
S
Q@IMGVF_kernel.buffer = internal unnamed_addr global [256 x float] undef, align 16
#i3280B

	full_text	

i32 0
%i3280B

	full_text
	
i32 128
0i6480B%
#
	full_text

i64 -1099511627776
$i3280B

	full_text


i32 32
{[3321 x float]*80Bd
b
	full_textU
S
Q@IMGVF_kernel.IMGVF = internal unnamed_addr global [3321 x float] undef, align 16
2float80B%
#
	full_text

float 1.000000e+00
$i6480B

	full_text


i64 32
'i3280B

	full_text

	i32 undef       	  
 

                      !    "# "" $% $( '' )* )) +, +- ++ ./ .0 .. 12 13 11 45 47 68 66 9: 9; 99 <= <> << ?@ ?? AB AC AA DE DD FG FF HI HH JK JJ LM LN LL OQ PP RS RT RR UV UX WW YY Z[ ZZ \] \^ _` ab aa cd cc ef ee gg hh ij ii kk lm ln ll op or qs qq tu tv tt wx ww yz yy {| {{ }~ }} €  ‚  ƒ
„ ƒƒ …† …… ‡ˆ ‡‡ ‰Š ‰‰ ‹‹ ŒŒ  Ž ŽŽ  ‘’ ‘‘ “” ““ •– •• —˜ —— ™š ™™ ›œ ›› 
ž  Ÿ  ŸŸ ¡¢ ¡¡ £¤ ££ ¥¦ ¥¥ §
¨ §§ ©ª ©© «¬ «« ­® ­­ ¯° ¯¯ ±
² ±± ³´ ³³ µ¶ µµ ·¸ ·· ¹º ¹¹ »
¼ »» ½¾ ½½ ¿À ¿¿ ÁÂ ÁÁ ÃÄ ÃÃ Å
Æ ÅÅ ÇÈ ÇÇ ÉÊ ÉÉ ËÌ ËË ÍÎ ÍÍ Ï
Ð ÏÏ ÑÒ ÑÑ ÓÔ ÓÓ ÕÖ ÕÕ ×Ø ×× Ù
Ú ÙÙ ÛÜ ÛÛ ÝÞ ÝÝ ßà ßß áâ áá ã
ä ãã åç æ
è ææ é
ê éé ëì ëï îî ðñ ð
ò ðð óô ó
õ óó ö÷ öö øù øø úû ú
ü úú ýþ ýý ÿ€ ÿ
 ÿÿ ‚ƒ ‚‚ „… „
† „„ ‡ˆ ‡
‰ ‡‡ Š‹ Š
Œ ŠŠ Ž 
  ‘ 
’  “” “– •• —˜ —— ™š ™
› ™™ œ œ
ž œœ Ÿ  ŸŸ ¡¢ ¡
£ ¡
¤ ¡¡ ¥¦ ¥¥ §¨ §§ ©ª ©
« ©© ¬­ ¬
® ¬¬ ¯° ¯¯ ±² ±
³ ±
´ ±± µ¶ µ
· µµ ¸¹ ¸
º ¸¸ »¼ »» ½
¾ ½½ ¿À ¿¿ ÁÂ Á
Ã ÁÁ ÄÅ Ä
Æ ÄÄ ÇÈ ÇÇ É
Ê ÉÉ ËÌ ËË ÍÎ Í
Ï ÍÍ ÐÑ Ð
Ò ÐÐ ÓÔ Ó
Õ ÓÓ Ö× ÖÖ Ø
Ù ØØ ÚÛ ÚÚ ÜÝ Ü
Þ ÜÜ ßà ß
á ßß âã ââ ä
å ää æç ææ èé è
ê èè ëì ë
í ëë îï îî ð
ñ ðð òó òò ôõ ô
ö ôô ÷ø ÷
ù ÷÷ úû úú ü
ý üü þÿ þþ € €
‚ €€ ƒ„ ƒ
… ƒƒ †‡ †† ˆ
‰ ˆˆ Š‹ ŠŠ Œ Œ
Ž ŒŒ  
‘  ’“ ’’ ”
• ”” –— –– ˜™ ˜
š ˜˜ ›œ ›
 ›› žŸ žž  
¡    ¢£ ¢¢ ¤¥ ¤
¦ ¤¤ §¨ §§ ©ª ©
« ©© ¬
­ ¬¬ ®¯ ®® °± °° ²³ ²
´ ²² µ¶ µµ ·¸ ·
¹ ·· º» º
¼ ºº ½¾ ½½ ¿À ¿¿ ÁÂ Á
Ã ÁÁ ÄÅ ÄÄ ÆÇ Æ
È ÆÆ ÉÊ É
Ë ÉÉ ÌÍ ÌÌ ÎÏ Î
Ð ÎÎ ÑÒ Ñ
Ó ÑÑ ÔÕ ÔÔ Ö× Ö
Ø ÖÖ ÙÚ Ù
Û ÙÙ ÜÝ ÜÜ Þß Þ
à ÞÞ áâ á
ã áá äå ää æç æ
è ææ éê é
ë é
ì éé íî í
ï í
ð íí ñò ñ
ó ñ
ô ññ õö õ
÷ õ
ø õõ ùú ù
û ù
ü ùù ýþ ý
ÿ ý
€ ýý ‚ 
ƒ 
„  …† …
‡ …… ˆ‰ ˆ
Š ˆˆ ‹Œ ‹‹ Ž   
‘  ’
“ ’’ ”• ”
– ”
— ”” ˜š ™™ ›œ ›› ž  Ÿ  Ÿ
¡ ŸŸ ¢£ ¢
¤ ¢¢ ¥¦ ¥¨ §§ ©ª ©
« ©© ¬­ ¬
® ¬¬ ¯° ¯¯ ±
² ±± ³´ ³³ µ¶ µ
· µµ ¸º ¹
» ¹¹ ¼½ ¼¿ ¾Á À
Â ÀÀ ÃÄ Ã
Å ÃÃ ÆÇ ÆÆ È
É ÈÈ ÊÌ Ë
Í ËË ÎÏ Î
Ð ÎÎ ÑÓ Ò
Ô ÒÒ ÕÖ ÕÕ ×Ø ×
Ù ×× ÚÚ ÛÜ ÛÛ ÝÞ Ý
ß ÝÝ àá à
ã ââ äå ä
æ ää çè ç
é çç êê ëì ëî íí ïð ïï ñò ñ
ó ññ ôõ ô
ö ôô ÷ø ùú ùü ûû ýþ ýý ÿ€ ÿ
 ÿÿ ‚ƒ ‚
„ ‚‚ …† ‡ˆ ‡Š ‰‰ ‹Œ ‹
 ‹‹ Ž ŽŽ ‘ ’ “” •– •• —— ˜™ ˜˜ š› šš œ œ
ž œœ Ÿ  Ÿ¢ ¡¥ ¤¤ ¦§ ¦¦ ¨© ¨
ª ¨¨ «¬ «
­ «« ®¯ ®
° ®® ±² ±´ ³
µ ³³ ¶· ¶
¸ ¶¶ ¹º ¹
» ¹¹ ¼½ ¼¼ ¾
¿ ¾¾ ÀÁ ÀÀ ÂÃ ÂÂ ÄÅ Ä
Æ ÄÄ ÇÈ ÇÇ ÉÊ É
Ë ÉÉ ÌÎ ÍÍ ÏÐ Ï
Ñ ÏÏ ÒÓ ÒÖ ÕÕ ×Ø ×× ÙÚ Ù
Û ÙÙ ÜÝ Ü
Þ ÜÜ ßà áâ áä ãã åæ åå çè ç
é çç êë ê
ì êê íî ïð ïò ññ óô óó õö õ
÷ õõ øù ø
ú øø ûü ýþ ý€ ÿÿ ‚  ƒ„ ƒ
… ƒƒ †‡ †
ˆ †† ‰Š ‹Œ ‹Ž    ‘’ ‘
“ ‘‘ ”• ”
– ”” —˜ ™š ™œ ›› ž  Ÿ  Ÿ
¡ ŸŸ ¢£ ¢
¤ ¢¢ ¥¦ §¨ §ª ©© «¬ «« ­® ­
¯ ­­ °± °
² °° ³´ µ¶ µ· 
¸ Ž¹ k
¹ šº » 
¼ ‹¼ Œ¼ 
¼ 
¼ ¿½ 	¾ g¿ 
À Œ
À 
À ŽÀ 
À §
À °    	 
 
          ! #" %P (' *) ,  -+ / 0. 2 31 5+ 7 8. : ;6 =9 >< @ B? CA ED G? IH KF MJ N' QP S TR V. X  [Z ] ba d fh jk mi nl p  r sq ue v xw zy |  ~w € ‚ „ † ˆ Š‹ { ’ ”  – ˜— š™ œ› ž    ¢¡ ¤£ ¦¥ ¨  ª ¬« ®­ °¯ ²  ´ ¶µ ¸· º¹ ¼  ¾ À¿ ÂÁ ÄÃ Æ  È ÊÉ ÌË ÎÍ Ð  Ò ÔÓ ÖÕ Ø× Ú  Ü ÞÝ àß âá äW çä è• ê" ìÛ ï‚ ñæ ò ôt õ× ÷î ùø û  üú þc €ý ÿ ƒó …e †„ ˆ ‰‡ ‹ Œ„ ŽŠ  ‘‚ ’ ”‚ –‚ ˜• š— ›‡ ‚ ž‚  œ ¢‡ £Ÿ ¤ ¦ ¨¥ ª§ « ­‰ ® °¬ ²‰ ³¯ ´ ¶‚ ·µ ¹ º¸ ¼» ¾½ À™ Â ÃÁ Å ÆÄ ÈÇ ÊÉ ÌË Î¿ Ï¡ Ñ ÒÐ Ô ÕÓ ×Ö ÙØ ÛÚ Ý¿ Þµ à© áß ãâ åä çæ é¿ êµ ì± íë ïî ñð óò õ¿ öÁ ø± ù÷ ûú ýü ÿþ ¿ ‚Ð „± …ƒ ‡† ‰ˆ ‹Š ¿ ŽÁ © ‘ “’ •” —– ™¿ šÐ œ© › Ÿž ¡  £¢ ¥¿ ¦Í ¨g ª§ «© ­¬ ¯Ü ±g ³° ´² ¶è ¸‹ ¹g »· ¼º ¾ô Àg Â¿ ÃÁ ÅŒ Ç€ Èg ÊÆ ËÉ Í ÏŒ Ðg ÒÎ ÓÑ ÕŽ ×˜ Øg ÚÖ ÛÙ Ý ß¤ àg âÞ ãá åÜ çµ è® êÍ ëæ ì½ îè ïé ðÄ òô óí ôÌ ö€ ÷ñ øÔ úŒ ûõ üÜ þ˜ ÿù €ä ‚¤ ƒý „ †¿ ‡ ‰» Šˆ Œ‹ Ž… ‹ ‘ “’ • –… —¿ š” œî žð   ¡Ÿ £ ¤¢ ¦‘ ¨ð ª «© ­ó ®¬ °¯ ²± ´§ ¶³ ·î º“ »¹ ½ ¿ Á‚ ÂÀ Ä ÅÃ ÇÆ ÉÈ Ì{ Í› ÏË Ð› Ó™ ÔÒ Öö ØÕ Ùî ÜÛ Þ ßÝ á× ãæ å‚ æâ è{ é} ì{ îƒ ðí òï óñ õƒ ö• ú ü{ þû €ý ÿ ƒ{ „Ÿ ˆ{ Š‰ Œ… ‹ Ž ‘é –— ™• ›š ˜ žœ  " ¢Í ¥¤ §¦ ©  ª¨ ¬ ­« ¯ °® ²¨ ´ µ« · ¸³ º¶ »¹ ½¼ ¿¾ ÁÀ Ã Å¼ ÆÄ ÈÂ ÊÇ Ë¤ ÎÍ Ð ÑÏ Ó§ Ö{ ØÕ Ú× ÛÙ Ý{ Þ© â± ä{ æã èå éç ë{ ì³ ð» ò{ ôñ öó ÷õ ù{ ú½ þÅ €{ ‚ÿ „ …ƒ ‡{ ˆÇ ŒÏ Ž{  ’ “‘ •{ –Ñ šÙ œ{ ž›   ¡Ÿ £{ ¤Û ¨ã ª{ ¬© ®« ¯­ ±{ ²Z ¶$ &$ W& '\ ^\ `4 64 P_ `o qo ¡O PU 'U Wå æ¡ £¡ Ôë íë â£ ¤í îë íë ø± ³± Í“ •“ ™÷ øù ûù †Ì ÍÒ ¤Ò Ô˜ ™¥ §¥ ¹… †‡ Õ‡ à¸ ¹¼ Ë¼ ¾ß àá ãá îÑ Ò¾ À¾ Òí îï ñï üà îà âÊ Ëû üý ÿý Š‰ Š‹ ‹ ˜— ˜™ ›™ ¦¥ ¦§ ©§ ´³ ´µ ‰µ ” ’ ”Ÿ æŸ ¡“ ” ÃÃ ÅÅ ÆÆ Ô ÁÁ ÂÂ ÄÄ” ÅÅ ”Õ ÆÆ Õ® ÄÄ ®† ÃÃ †ê ÃÃ êà ÃÃ àõ ÅÅ õ` ÃÃ `… ÅÅ …é ÅÅ éÄ ÄÄ Äî ÃÃ îÜ ÄÄ Üµ ÄÄ µñ ÅÅ ñ ÁÁ ý ÅÅ ýÚ ÃÃ Úí ÅÅ í ÅÅ ½ ÄÄ ½ù ÅÅ ù ÂÂ Š ÃÃ Š˜ ÃÃ ˜” ÃÃ ”¦ ÃÃ ¦Ì ÄÄ Ìø ÃÃ ø´ ÃÃ ´Y ÃÃ YÔ ÄÄ Ôä ÄÄ äü ÃÃ ü
Ç Õ
È 
É Ç
Ê Ë	Ë 	Ë }	Ì Ì e
Í Ñ	Î H	Î {
Î ƒ
Î 
Î §
Î ±
Î »
Î Å
Î Ï
Î Ù
Î ã
Î ½
Î É
Î Ø
Î ä
Î ð
Î ü
Î ˆ
Î ”
Î  
Î ±
Î È
Î ¾	Ï PÏ YÏ `
Ï Û
Ï Ÿ
Ï ¯Ï Ú
Ï ÛÏ êÏ øÏ †Ï ’Ï ”
Ï •
Ï ÍÏ àÏ îÏ üÏ ŠÏ ˜Ï ¦Ï ´
Ð Á
Ñ ß
Ò ³	Ó ^Ó h
Ó ’Ó —	Ô )
Ô ½
Ô ø
Ô ¦
Õ ­
Ö Ÿ
× ·
Ø ö
Ø ™
Ø ›Ø â
Ù …
Ú £
Û ™Ü ‹Ü ¬Ü ’
Ý ‡
Ý ‰
Ý “
Ý —
Ý §Þ {Þ ƒÞ Þ §Þ ±Þ »Þ ÅÞ ÏÞ ÙÞ ãß ß 	ß "	ß '	ß Zß ^	ß i	ß kß é
ß î
ß Š
ß •
ß ™
ß ¥
ß ©
ß 
ß ˜
ß ¤
à •	á 
â ©ã Hã ½ã Éã Øã äã ðã üã ˆã ”ã  ã ±ã Èã ¾ä cä g	å 	å 	å w	å y
å 
å —
å ›
å ¡
å ¥
å «
å ¯
å µ
å ¹
å ¿
å Ã
å É
å Í
å Ó
å ×
å Ý
å áæ W"
IMGVF_kernel"
_Z12get_group_idj"
_Z12get_local_idj"
_Z7barrierj"
	heaviside"
llvm.fmuladd.f32"

_Z4fabsf*ž
%rodinia-3.1-leukocyte-IMGVF_kernel.clu
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
 
transfer_bytes_log1p
G2‚A

wgsize_log1p
G2‚A

wgsize
€

transfer_bytes
„ÜÉ