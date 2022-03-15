

[external]
JcallBB
@
	full_text3
1
/%7 = tail call i64 @_Z12get_group_idj(i32 0) #5
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
JcallBB
@
	full_text3
1
/%9 = tail call i64 @_Z12get_local_idj(i32 0) #5
.shlB'
%
	full_text

%10 = shl i64 %7, 32
"i64B

	full_text


i64 %7
7ashrB/
-
	full_text 

%11 = ashr exact i64 %10, 32
#i64B

	full_text
	
i64 %10
sgetelementptrBb
`
	full_textS
Q
O%12 = getelementptr inbounds %struct.dim_str, %struct.dim_str* %1, i64 0, i32 4
FloadB>
<
	full_text/
-
+%13 = load i64, i64* %12, align 8, !tbaa !9
%i64*B

	full_text


i64* %12
6icmpB.
,
	full_text

%14 = icmp slt i64 %11, %13
#i64B

	full_text
	
i64 %11
#i64B

	full_text
	
i64 %13
9brB3
1
	full_text$
"
 br i1 %14, label %15, label %219
!i1B

	full_text


i1 %14
7trunc8B,
*
	full_text

%16 = trunc i64 %9 to i32
$i648B

	full_text


i64 %9
ugetelementptr8Bb
`
	full_textS
Q
O%17 = getelementptr inbounds %struct.par_str, %struct.par_str* %0, i64 0, i32 0
Mload8BC
A
	full_text4
2
0%18 = load float, float* %17, align 4, !tbaa !15
+float*8B

	full_text


float* %17
?fmul8B5
3
	full_text&
$
"%19 = fmul float %18, 2.000000e+00
)float8B

	full_text

	float %18
6fmul8B,
*
	full_text

%20 = fmul float %18, %19
)float8B

	full_text

	float %18
)float8B

	full_text

	float %19
wgetelementptr8Bd
b
	full_textU
S
Q%21 = getelementptr inbounds %struct.box_str, %struct.box_str* %2, i64 %11, i32 4
%i648B

	full_text
	
i64 %11
Iload8B?
=
	full_text0
.
,%22 = load i64, i64* %21, align 8, !tbaa !18
'i64*8B

	full_text


i64* %21
8icmp8B.
,
	full_text

%23 = icmp slt i32 %16, 100
%i328B

	full_text
	
i32 %16
:br8B2
0
	full_text#
!
br i1 %23, label %24, label %80
#i18B

	full_text


i1 %23
0shl8B'
%
	full_text

%25 = shl i64 %9, 32
$i648B

	full_text


i64 %9
9ashr8B/
-
	full_text 

%26 = ashr exact i64 %25, 32
%i648B

	full_text
	
i64 %25
1shl8B(
&
	full_text

%27 = shl i64 %22, 32
%i648B

	full_text
	
i64 %22
9ashr8B/
-
	full_text 

%28 = ashr exact i64 %27, 32
%i648B

	full_text
	
i64 %27
8icmp8B.
,
	full_text

%29 = icmp sgt i64 %26, -28
%i648B

	full_text
	
i64 %26
Dselect8B8
6
	full_text)
'
%%30 = select i1 %29, i64 %26, i64 -28
#i18B

	full_text


i1 %29
%i648B

	full_text
	
i64 %26
6add8B-
+
	full_text

%31 = add nsw i64 %30, 127
%i648B

	full_text
	
i64 %30
6sub8B-
+
	full_text

%32 = sub nsw i64 %31, %26
%i648B

	full_text
	
i64 %31
%i648B

	full_text
	
i64 %26
2lshr8B(
&
	full_text

%33 = lshr i64 %32, 7
%i648B

	full_text
	
i64 %32
8add8B/
-
	full_text 

%34 = add nuw nsw i64 %33, 1
%i648B

	full_text
	
i64 %33
0and8B'
%
	full_text

%35 = and i64 %34, 3
%i648B

	full_text
	
i64 %34
5icmp8B+
)
	full_text

%36 = icmp eq i64 %35, 0
%i648B

	full_text
	
i64 %35
:br8B2
0
	full_text#
!
br i1 %36, label %49, label %37
#i18B

	full_text


i1 %36
'br8B

	full_text

br label %38
Dphi8B;
9
	full_text,
*
(%39 = phi i64 [ %26, %37 ], [ %46, %38 ]
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %46
Dphi8B;
9
	full_text,
*
(%40 = phi i64 [ %35, %37 ], [ %47, %38 ]
%i648B

	full_text
	
i64 %35
%i648B

	full_text
	
i64 %47
Ægetelementptr8Bö
ó
	full_textâ
Ü
É%41 = getelementptr inbounds [100 x %struct.FOUR_VECTOR], [100 x %struct.FOUR_VECTOR]* @kernel_gpu_opencl.rA_shared, i64 0, i64 %39
%i648B

	full_text
	
i64 %39
6add8B-
+
	full_text

%42 = add nsw i64 %39, %28
%i648B

	full_text
	
i64 %39
%i648B

	full_text
	
i64 %28
xgetelementptr8Be
c
	full_textV
T
R%43 = getelementptr inbounds %struct.FOUR_VECTOR, %struct.FOUR_VECTOR* %3, i64 %42
%i648B

	full_text
	
i64 %42
Mbitcast8B@
>
	full_text1
/
-%44 = bitcast %struct.FOUR_VECTOR* %41 to i8*
-struct*8B

	full_text

struct* %41
Mbitcast8B@
>
	full_text1
/
-%45 = bitcast %struct.FOUR_VECTOR* %43 to i8*
-struct*8B

	full_text

struct* %43
ucall8Bk
i
	full_text\
Z
Xcall void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %44, i8* align 4 %45, i64 16, i1 false)
%i8*8B

	full_text
	
i8* %44
%i8*8B

	full_text
	
i8* %45
6add8B-
+
	full_text

%46 = add nsw i64 %39, 128
%i648B

	full_text
	
i64 %39
1add8B(
&
	full_text

%47 = add i64 %40, -1
%i648B

	full_text
	
i64 %40
5icmp8B+
)
	full_text

%48 = icmp eq i64 %47, 0
%i648B

	full_text
	
i64 %47
Jbr8BB
@
	full_text3
1
/br i1 %48, label %49, label %38, !llvm.loop !20
#i18B

	full_text


i1 %48
Dphi8B;
9
	full_text,
*
(%50 = phi i64 [ %26, %24 ], [ %46, %38 ]
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %46
8icmp8B.
,
	full_text

%51 = icmp ult i64 %32, 384
%i648B

	full_text
	
i64 %32
:br8B2
0
	full_text#
!
br i1 %51, label %80, label %52
#i18B

	full_text


i1 %51
'br8B

	full_text

br label %53
Dphi8B;
9
	full_text,
*
(%54 = phi i64 [ %50, %52 ], [ %78, %53 ]
%i648B

	full_text
	
i64 %50
%i648B

	full_text
	
i64 %78
Ægetelementptr8Bö
ó
	full_textâ
Ü
É%55 = getelementptr inbounds [100 x %struct.FOUR_VECTOR], [100 x %struct.FOUR_VECTOR]* @kernel_gpu_opencl.rA_shared, i64 0, i64 %54
%i648B

	full_text
	
i64 %54
6add8B-
+
	full_text

%56 = add nsw i64 %54, %28
%i648B

	full_text
	
i64 %54
%i648B

	full_text
	
i64 %28
xgetelementptr8Be
c
	full_textV
T
R%57 = getelementptr inbounds %struct.FOUR_VECTOR, %struct.FOUR_VECTOR* %3, i64 %56
%i648B

	full_text
	
i64 %56
Mbitcast8B@
>
	full_text1
/
-%58 = bitcast %struct.FOUR_VECTOR* %55 to i8*
-struct*8B

	full_text

struct* %55
Mbitcast8B@
>
	full_text1
/
-%59 = bitcast %struct.FOUR_VECTOR* %57 to i8*
-struct*8B

	full_text

struct* %57
ucall8Bk
i
	full_text\
Z
Xcall void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %58, i8* align 4 %59, i64 16, i1 false)
%i8*8B

	full_text
	
i8* %58
%i8*8B

	full_text
	
i8* %59
6add8B-
+
	full_text

%60 = add nsw i64 %54, 128
%i648B

	full_text
	
i64 %54
Ægetelementptr8Bö
ó
	full_textâ
Ü
É%61 = getelementptr inbounds [100 x %struct.FOUR_VECTOR], [100 x %struct.FOUR_VECTOR]* @kernel_gpu_opencl.rA_shared, i64 0, i64 %60
%i648B

	full_text
	
i64 %60
6add8B-
+
	full_text

%62 = add nsw i64 %60, %28
%i648B

	full_text
	
i64 %60
%i648B

	full_text
	
i64 %28
xgetelementptr8Be
c
	full_textV
T
R%63 = getelementptr inbounds %struct.FOUR_VECTOR, %struct.FOUR_VECTOR* %3, i64 %62
%i648B

	full_text
	
i64 %62
Mbitcast8B@
>
	full_text1
/
-%64 = bitcast %struct.FOUR_VECTOR* %61 to i8*
-struct*8B

	full_text

struct* %61
Mbitcast8B@
>
	full_text1
/
-%65 = bitcast %struct.FOUR_VECTOR* %63 to i8*
-struct*8B

	full_text

struct* %63
ucall8Bk
i
	full_text\
Z
Xcall void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %64, i8* align 4 %65, i64 16, i1 false)
%i8*8B

	full_text
	
i8* %64
%i8*8B

	full_text
	
i8* %65
6add8B-
+
	full_text

%66 = add nsw i64 %54, 256
%i648B

	full_text
	
i64 %54
Ægetelementptr8Bö
ó
	full_textâ
Ü
É%67 = getelementptr inbounds [100 x %struct.FOUR_VECTOR], [100 x %struct.FOUR_VECTOR]* @kernel_gpu_opencl.rA_shared, i64 0, i64 %66
%i648B

	full_text
	
i64 %66
6add8B-
+
	full_text

%68 = add nsw i64 %66, %28
%i648B

	full_text
	
i64 %66
%i648B

	full_text
	
i64 %28
xgetelementptr8Be
c
	full_textV
T
R%69 = getelementptr inbounds %struct.FOUR_VECTOR, %struct.FOUR_VECTOR* %3, i64 %68
%i648B

	full_text
	
i64 %68
Mbitcast8B@
>
	full_text1
/
-%70 = bitcast %struct.FOUR_VECTOR* %67 to i8*
-struct*8B

	full_text

struct* %67
Mbitcast8B@
>
	full_text1
/
-%71 = bitcast %struct.FOUR_VECTOR* %69 to i8*
-struct*8B

	full_text

struct* %69
ucall8Bk
i
	full_text\
Z
Xcall void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %70, i8* align 4 %71, i64 16, i1 false)
%i8*8B

	full_text
	
i8* %70
%i8*8B

	full_text
	
i8* %71
6add8B-
+
	full_text

%72 = add nsw i64 %54, 384
%i648B

	full_text
	
i64 %54
Ægetelementptr8Bö
ó
	full_textâ
Ü
É%73 = getelementptr inbounds [100 x %struct.FOUR_VECTOR], [100 x %struct.FOUR_VECTOR]* @kernel_gpu_opencl.rA_shared, i64 0, i64 %72
%i648B

	full_text
	
i64 %72
6add8B-
+
	full_text

%74 = add nsw i64 %72, %28
%i648B

	full_text
	
i64 %72
%i648B

	full_text
	
i64 %28
xgetelementptr8Be
c
	full_textV
T
R%75 = getelementptr inbounds %struct.FOUR_VECTOR, %struct.FOUR_VECTOR* %3, i64 %74
%i648B

	full_text
	
i64 %74
Mbitcast8B@
>
	full_text1
/
-%76 = bitcast %struct.FOUR_VECTOR* %73 to i8*
-struct*8B

	full_text

struct* %73
Mbitcast8B@
>
	full_text1
/
-%77 = bitcast %struct.FOUR_VECTOR* %75 to i8*
-struct*8B

	full_text

struct* %75
ucall8Bk
i
	full_text\
Z
Xcall void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %76, i8* align 4 %77, i64 16, i1 false)
%i8*8B

	full_text
	
i8* %76
%i8*8B

	full_text
	
i8* %77
6add8B-
+
	full_text

%78 = add nsw i64 %54, 512
%i648B

	full_text
	
i64 %54
9icmp8B/
-
	full_text 

%79 = icmp slt i64 %54, -412
%i648B

	full_text
	
i64 %54
:br8B2
0
	full_text#
!
br i1 %79, label %53, label %80
#i18B

	full_text


i1 %79
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #6
wgetelementptr8Bd
b
	full_textU
S
Q%81 = getelementptr inbounds %struct.box_str, %struct.box_str* %2, i64 %11, i32 5
%i648B

	full_text
	
i64 %11
Iload8B?
=
	full_text0
.
,%82 = load i32, i32* %81, align 8, !tbaa !22
'i32*8B

	full_text


i32* %81
6icmp8B,
*
	full_text

%83 = icmp slt i32 %82, 0
%i328B

	full_text
	
i32 %82
;br8B3
1
	full_text$
"
 br i1 %83, label %219, label %84
#i18B

	full_text


i1 %83
0shl8	B'
%
	full_text

%85 = shl i64 %9, 32
$i648	B

	full_text


i64 %9
9ashr8	B/
-
	full_text 

%86 = ashr exact i64 %85, 32
%i648	B

	full_text
	
i64 %85
1shl8	B(
&
	full_text

%87 = shl i64 %22, 32
%i648	B

	full_text
	
i64 %22
9ashr8	B/
-
	full_text 

%88 = ashr exact i64 %87, 32
%i648	B

	full_text
	
i64 %87
8icmp8	B.
,
	full_text

%89 = icmp sgt i64 %86, -28
%i648	B

	full_text
	
i64 %86
Dselect8	B8
6
	full_text)
'
%%90 = select i1 %89, i64 %86, i64 -28
#i18	B

	full_text


i1 %89
%i648	B

	full_text
	
i64 %86
6add8	B-
+
	full_text

%91 = add nsw i64 %90, 127
%i648	B

	full_text
	
i64 %90
6sub8	B-
+
	full_text

%92 = sub nsw i64 %91, %86
%i648	B

	full_text
	
i64 %91
%i648	B

	full_text
	
i64 %86
2lshr8	B(
&
	full_text

%93 = lshr i64 %92, 7
%i648	B

	full_text
	
i64 %92
0and8	B'
%
	full_text

%94 = and i64 %93, 1
%i648	B

	full_text
	
i64 %93
5icmp8	B+
)
	full_text

%95 = icmp eq i64 %94, 0
%i648	B

	full_text
	
i64 %94
Ægetelementptr8	Bö
ó
	full_textâ
Ü
É%96 = getelementptr inbounds [100 x %struct.FOUR_VECTOR], [100 x %struct.FOUR_VECTOR]* @kernel_gpu_opencl.rB_shared, i64 0, i64 %86
%i648	B

	full_text
	
i64 %86
Mbitcast8	B@
>
	full_text1
/
-%97 = bitcast %struct.FOUR_VECTOR* %96 to i8*
-struct*8	B

	full_text

struct* %96
çgetelementptr8	Bz
x
	full_textk
i
g%98 = getelementptr inbounds [100 x float], [100 x float]* @kernel_gpu_opencl.qB_shared, i64 0, i64 %86
%i648	B

	full_text
	
i64 %86
@bitcast8	B3
1
	full_text$
"
 %99 = bitcast float* %98 to i32*
+float*8	B

	full_text


float* %98
7add8	B.
,
	full_text

%100 = add nsw i64 %86, 128
%i648	B

	full_text
	
i64 %86
6icmp8	B,
*
	full_text

%101 = icmp eq i64 %93, 0
%i648	B

	full_text
	
i64 %93
(br8	B 

	full_text

br label %102
Ephi8
B<
:
	full_text-
+
)%103 = phi i64 [ %215, %214 ], [ 0, %84 ]
&i648
B

	full_text


i64 %215
7icmp8
B-
+
	full_text

%104 = icmp eq i64 %103, 0
&i648
B

	full_text


i64 %103
=br8
B5
3
	full_text&
$
"br i1 %104, label %109, label %105
$i18
B

	full_text
	
i1 %104
7add8B.
,
	full_text

%106 = add nsw i64 %103, -1
&i648B

	full_text


i64 %103
âgetelementptr8Bv
t
	full_textg
e
c%107 = getelementptr inbounds %struct.box_str, %struct.box_str* %2, i64 %11, i32 6, i64 %106, i32 3
%i648B

	full_text
	
i64 %11
&i648B

	full_text


i64 %106
Kload8BA
?
	full_text2
0
.%108 = load i32, i32* %107, align 4, !tbaa !23
(i32*8B

	full_text

	i32* %107
(br8B 

	full_text

br label %109
Gphi8B>
<
	full_text/
-
+%110 = phi i32 [ %108, %105 ], [ %8, %102 ]
&i328B

	full_text


i32 %108
$i328B

	full_text


i32 %8
<br8B4
2
	full_text%
#
!br i1 %23, label %111, label %152
#i18B

	full_text


i1 %23
8sext8B.
,
	full_text

%112 = sext i32 %110 to i64
&i328B

	full_text


i32 %110
ygetelementptr8Bf
d
	full_textW
U
S%113 = getelementptr inbounds %struct.box_str, %struct.box_str* %2, i64 %112, i32 4
&i648B

	full_text


i64 %112
Kload8BA
?
	full_text2
0
.%114 = load i64, i64* %113, align 8, !tbaa !18
(i64*8B

	full_text

	i64* %113
3shl8B*
(
	full_text

%115 = shl i64 %114, 32
&i648B

	full_text


i64 %114
;ashr8B1
/
	full_text"
 
%116 = ashr exact i64 %115, 32
&i648B

	full_text


i64 %115
<br8B4
2
	full_text%
#
!br i1 %95, label %117, label %124
#i18B

	full_text


i1 %95
8add8B/
-
	full_text 

%118 = add nsw i64 %86, %116
%i648B

	full_text
	
i64 %86
&i648B

	full_text


i64 %116
zgetelementptr8Bg
e
	full_textX
V
T%119 = getelementptr inbounds %struct.FOUR_VECTOR, %struct.FOUR_VECTOR* %3, i64 %118
&i648B

	full_text


i64 %118
Obitcast8BB
@
	full_text3
1
/%120 = bitcast %struct.FOUR_VECTOR* %119 to i8*
.struct*8B

	full_text

struct* %119
vcall8Bl
j
	full_text]
[
Ycall void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %97, i8* align 4 %120, i64 16, i1 false)
%i8*8B

	full_text
	
i8* %97
&i8*8B

	full_text


i8* %120
^getelementptr8BK
I
	full_text<
:
8%121 = getelementptr inbounds float, float* %4, i64 %118
&i648B

	full_text


i64 %118
Bbitcast8B5
3
	full_text&
$
"%122 = bitcast float* %121 to i32*
,float*8B

	full_text

float* %121
Kload8BA
?
	full_text2
0
.%123 = load i32, i32* %122, align 4, !tbaa !25
(i32*8B

	full_text

	i32* %122
Jstore8B?
=
	full_text0
.
,store i32 %123, i32* %99, align 4, !tbaa !25
&i328B

	full_text


i32 %123
'i32*8B

	full_text


i32* %99
(br8B 

	full_text

br label %124
Hphi8B?
=
	full_text0
.
,%125 = phi i64 [ %100, %117 ], [ %86, %111 ]
&i648B

	full_text


i64 %100
%i648B

	full_text
	
i64 %86
=br8B5
3
	full_text&
$
"br i1 %101, label %152, label %126
$i18B

	full_text
	
i1 %101
(br8B 

	full_text

br label %127
Iphi8B@
>
	full_text1
/
-%128 = phi i64 [ %125, %126 ], [ %150, %127 ]
&i648B

	full_text


i64 %125
&i648B

	full_text


i64 %150
∞getelementptr8Bú
ô
	full_textã
à
Ö%129 = getelementptr inbounds [100 x %struct.FOUR_VECTOR], [100 x %struct.FOUR_VECTOR]* @kernel_gpu_opencl.rB_shared, i64 0, i64 %128
&i648B

	full_text


i64 %128
9add8B0
.
	full_text!

%130 = add nsw i64 %128, %116
&i648B

	full_text


i64 %128
&i648B

	full_text


i64 %116
zgetelementptr8Bg
e
	full_textX
V
T%131 = getelementptr inbounds %struct.FOUR_VECTOR, %struct.FOUR_VECTOR* %3, i64 %130
&i648B

	full_text


i64 %130
Obitcast8BB
@
	full_text3
1
/%132 = bitcast %struct.FOUR_VECTOR* %129 to i8*
.struct*8B

	full_text

struct* %129
Obitcast8BB
@
	full_text3
1
/%133 = bitcast %struct.FOUR_VECTOR* %131 to i8*
.struct*8B

	full_text

struct* %131
wcall8Bm
k
	full_text^
\
Zcall void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %132, i8* align 4 %133, i64 16, i1 false)
&i8*8B

	full_text


i8* %132
&i8*8B

	full_text


i8* %133
^getelementptr8BK
I
	full_text<
:
8%134 = getelementptr inbounds float, float* %4, i64 %130
&i648B

	full_text


i64 %130
Bbitcast8B5
3
	full_text&
$
"%135 = bitcast float* %134 to i32*
,float*8B

	full_text

float* %134
Kload8BA
?
	full_text2
0
.%136 = load i32, i32* %135, align 4, !tbaa !25
(i32*8B

	full_text

	i32* %135
ègetelementptr8B|
z
	full_textm
k
i%137 = getelementptr inbounds [100 x float], [100 x float]* @kernel_gpu_opencl.qB_shared, i64 0, i64 %128
&i648B

	full_text


i64 %128
Bbitcast8B5
3
	full_text&
$
"%138 = bitcast float* %137 to i32*
,float*8B

	full_text

float* %137
Kstore8B@
>
	full_text1
/
-store i32 %136, i32* %138, align 4, !tbaa !25
&i328B

	full_text


i32 %136
(i32*8B

	full_text

	i32* %138
8add8B/
-
	full_text 

%139 = add nsw i64 %128, 128
&i648B

	full_text


i64 %128
∞getelementptr8Bú
ô
	full_textã
à
Ö%140 = getelementptr inbounds [100 x %struct.FOUR_VECTOR], [100 x %struct.FOUR_VECTOR]* @kernel_gpu_opencl.rB_shared, i64 0, i64 %139
&i648B

	full_text


i64 %139
9add8B0
.
	full_text!

%141 = add nsw i64 %139, %116
&i648B

	full_text


i64 %139
&i648B

	full_text


i64 %116
zgetelementptr8Bg
e
	full_textX
V
T%142 = getelementptr inbounds %struct.FOUR_VECTOR, %struct.FOUR_VECTOR* %3, i64 %141
&i648B

	full_text


i64 %141
Obitcast8BB
@
	full_text3
1
/%143 = bitcast %struct.FOUR_VECTOR* %140 to i8*
.struct*8B

	full_text

struct* %140
Obitcast8BB
@
	full_text3
1
/%144 = bitcast %struct.FOUR_VECTOR* %142 to i8*
.struct*8B

	full_text

struct* %142
wcall8Bm
k
	full_text^
\
Zcall void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %143, i8* align 4 %144, i64 16, i1 false)
&i8*8B

	full_text


i8* %143
&i8*8B

	full_text


i8* %144
^getelementptr8BK
I
	full_text<
:
8%145 = getelementptr inbounds float, float* %4, i64 %141
&i648B

	full_text


i64 %141
Bbitcast8B5
3
	full_text&
$
"%146 = bitcast float* %145 to i32*
,float*8B

	full_text

float* %145
Kload8BA
?
	full_text2
0
.%147 = load i32, i32* %146, align 4, !tbaa !25
(i32*8B

	full_text

	i32* %146
ègetelementptr8B|
z
	full_textm
k
i%148 = getelementptr inbounds [100 x float], [100 x float]* @kernel_gpu_opencl.qB_shared, i64 0, i64 %139
&i648B

	full_text


i64 %139
Bbitcast8B5
3
	full_text&
$
"%149 = bitcast float* %148 to i32*
,float*8B

	full_text

float* %148
Kstore8B@
>
	full_text1
/
-store i32 %147, i32* %149, align 4, !tbaa !25
&i328B

	full_text


i32 %147
(i32*8B

	full_text

	i32* %149
8add8B/
-
	full_text 

%150 = add nsw i64 %128, 256
&i648B

	full_text


i64 %128
;icmp8B1
/
	full_text"
 
%151 = icmp slt i64 %128, -156
&i648B

	full_text


i64 %128
=br8B5
3
	full_text&
$
"br i1 %151, label %127, label %152
$i18B

	full_text
	
i1 %151
Wphi8BN
L
	full_text?
=
;%153 = phi i1 [ false, %109 ], [ %23, %127 ], [ %23, %124 ]
#i18B

	full_text


i1 %23
#i18B

	full_text


i1 %23
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #6
=br8B5
3
	full_text&
$
"br i1 %153, label %154, label %214
$i18B

	full_text
	
i1 %153
(br8B 

	full_text

br label %155
Hphi8B?
=
	full_text0
.
,%156 = phi i64 [ %212, %211 ], [ %86, %154 ]
&i648B

	full_text


i64 %212
%i648B

	full_text
	
i64 %86
∑getelementptr8B£
†
	full_textí
è
å%157 = getelementptr inbounds [100 x %struct.FOUR_VECTOR], [100 x %struct.FOUR_VECTOR]* @kernel_gpu_opencl.rA_shared, i64 0, i64 %156, i32 0
&i648B

	full_text


i64 %156
∑getelementptr8B£
†
	full_textí
è
å%158 = getelementptr inbounds [100 x %struct.FOUR_VECTOR], [100 x %struct.FOUR_VECTOR]* @kernel_gpu_opencl.rA_shared, i64 0, i64 %156, i32 1
&i648B

	full_text


i64 %156
∑getelementptr8B£
†
	full_textí
è
å%159 = getelementptr inbounds [100 x %struct.FOUR_VECTOR], [100 x %struct.FOUR_VECTOR]* @kernel_gpu_opencl.rA_shared, i64 0, i64 %156, i32 2
&i648B

	full_text


i64 %156
∑getelementptr8B£
†
	full_textí
è
å%160 = getelementptr inbounds [100 x %struct.FOUR_VECTOR], [100 x %struct.FOUR_VECTOR]* @kernel_gpu_opencl.rA_shared, i64 0, i64 %156, i32 3
&i648B

	full_text


i64 %156
8add8B/
-
	full_text 

%161 = add nsw i64 %156, %88
&i648B

	full_text


i64 %156
%i648B

	full_text
	
i64 %88
Ågetelementptr8Bn
l
	full_text_
]
[%162 = getelementptr inbounds %struct.FOUR_VECTOR, %struct.FOUR_VECTOR* %5, i64 %161, i32 0
&i648B

	full_text


i64 %161
Ågetelementptr8Bn
l
	full_text_
]
[%163 = getelementptr inbounds %struct.FOUR_VECTOR, %struct.FOUR_VECTOR* %5, i64 %161, i32 1
&i648B

	full_text


i64 %161
Ågetelementptr8Bn
l
	full_text_
]
[%164 = getelementptr inbounds %struct.FOUR_VECTOR, %struct.FOUR_VECTOR* %5, i64 %161, i32 2
&i648B

	full_text


i64 %161
Ågetelementptr8Bn
l
	full_text_
]
[%165 = getelementptr inbounds %struct.FOUR_VECTOR, %struct.FOUR_VECTOR* %5, i64 %161, i32 3
&i648B

	full_text


i64 %161
Oload8BE
C
	full_text6
4
2%166 = load float, float* %162, align 4, !tbaa !26
,float*8B

	full_text

float* %162
Oload8BE
C
	full_text6
4
2%167 = load float, float* %163, align 4, !tbaa !28
,float*8B

	full_text

float* %163
Oload8BE
C
	full_text6
4
2%168 = load float, float* %164, align 4, !tbaa !29
,float*8B

	full_text

float* %164
Oload8BE
C
	full_text6
4
2%169 = load float, float* %165, align 4, !tbaa !30
,float*8B

	full_text

float* %165
(br8B 

	full_text

br label %170
Kphi8BB
@
	full_text3
1
/%171 = phi float [ %169, %155 ], [ %208, %170 ]
*float8B

	full_text


float %169
*float8B

	full_text


float %208
Kphi8BB
@
	full_text3
1
/%172 = phi float [ %168, %155 ], [ %207, %170 ]
*float8B

	full_text


float %168
*float8B

	full_text


float %207
Kphi8BB
@
	full_text3
1
/%173 = phi float [ %167, %155 ], [ %206, %170 ]
*float8B

	full_text


float %167
*float8B

	full_text


float %206
Kphi8BB
@
	full_text3
1
/%174 = phi float [ %166, %155 ], [ %205, %170 ]
*float8B

	full_text


float %166
*float8B

	full_text


float %205
Fphi8B=
;
	full_text.
,
*%175 = phi i64 [ 0, %155 ], [ %209, %170 ]
&i648B

	full_text


i64 %209
Pload8BF
D
	full_text7
5
3%176 = load float, float* %157, align 16, !tbaa !26
,float*8B

	full_text

float* %157
∑getelementptr8B£
†
	full_textí
è
å%177 = getelementptr inbounds [100 x %struct.FOUR_VECTOR], [100 x %struct.FOUR_VECTOR]* @kernel_gpu_opencl.rB_shared, i64 0, i64 %175, i32 0
&i648B

	full_text


i64 %175
Pload8BF
D
	full_text7
5
3%178 = load float, float* %177, align 16, !tbaa !26
,float*8B

	full_text

float* %177
9fadd8B/
-
	full_text 

%179 = fadd float %176, %178
*float8B

	full_text


float %176
*float8B

	full_text


float %178
Oload8BE
C
	full_text6
4
2%180 = load float, float* %158, align 4, !tbaa !28
,float*8B

	full_text

float* %158
∑getelementptr8B£
†
	full_textí
è
å%181 = getelementptr inbounds [100 x %struct.FOUR_VECTOR], [100 x %struct.FOUR_VECTOR]* @kernel_gpu_opencl.rB_shared, i64 0, i64 %175, i32 1
&i648B

	full_text


i64 %175
Oload8BE
C
	full_text6
4
2%182 = load float, float* %181, align 4, !tbaa !28
,float*8B

	full_text

float* %181
Oload8BE
C
	full_text6
4
2%183 = load float, float* %159, align 8, !tbaa !29
,float*8B

	full_text

float* %159
∑getelementptr8B£
†
	full_textí
è
å%184 = getelementptr inbounds [100 x %struct.FOUR_VECTOR], [100 x %struct.FOUR_VECTOR]* @kernel_gpu_opencl.rB_shared, i64 0, i64 %175, i32 2
&i648B

	full_text


i64 %175
Oload8BE
C
	full_text6
4
2%185 = load float, float* %184, align 8, !tbaa !29
,float*8B

	full_text

float* %184
9fmul8B/
-
	full_text 

%186 = fmul float %183, %185
*float8B

	full_text


float %183
*float8B

	full_text


float %185
icall8B_
]
	full_textP
N
L%187 = tail call float @llvm.fmuladd.f32(float %180, float %182, float %186)
*float8B

	full_text


float %180
*float8B

	full_text


float %182
*float8B

	full_text


float %186
Oload8BE
C
	full_text6
4
2%188 = load float, float* %160, align 4, !tbaa !30
,float*8B

	full_text

float* %160
∑getelementptr8B£
†
	full_textí
è
å%189 = getelementptr inbounds [100 x %struct.FOUR_VECTOR], [100 x %struct.FOUR_VECTOR]* @kernel_gpu_opencl.rB_shared, i64 0, i64 %175, i32 3
&i648B

	full_text


i64 %175
Oload8BE
C
	full_text6
4
2%190 = load float, float* %189, align 4, !tbaa !30
,float*8B

	full_text

float* %189
icall8B_
]
	full_textP
N
L%191 = tail call float @llvm.fmuladd.f32(float %188, float %190, float %187)
*float8B

	full_text


float %188
*float8B

	full_text


float %190
*float8B

	full_text


float %187
9fsub8B/
-
	full_text 

%192 = fsub float %179, %191
*float8B

	full_text


float %179
*float8B

	full_text


float %191
8fmul8B.
,
	full_text

%193 = fmul float %20, %192
)float8B

	full_text

	float %20
*float8B

	full_text


float %192
Bfsub8B8
6
	full_text)
'
%%194 = fsub float -0.000000e+00, %193
*float8B

	full_text


float %193
Kcall8BA
?
	full_text2
0
.%195 = tail call float @_Z3expf(float %194) #5
*float8B

	full_text


float %194
Afmul8B7
5
	full_text(
&
$%196 = fmul float %195, 2.000000e+00
*float8B

	full_text


float %195
9fsub8B/
-
	full_text 

%197 = fsub float %180, %182
*float8B

	full_text


float %180
*float8B

	full_text


float %182
9fmul8B/
-
	full_text 

%198 = fmul float %197, %196
*float8B

	full_text


float %197
*float8B

	full_text


float %196
9fsub8B/
-
	full_text 

%199 = fsub float %183, %185
*float8B

	full_text


float %183
*float8B

	full_text


float %185
9fmul8B/
-
	full_text 

%200 = fmul float %199, %196
*float8B

	full_text


float %199
*float8B

	full_text


float %196
9fsub8B/
-
	full_text 

%201 = fsub float %188, %190
*float8B

	full_text


float %188
*float8B

	full_text


float %190
9fmul8B/
-
	full_text 

%202 = fmul float %201, %196
*float8B

	full_text


float %201
*float8B

	full_text


float %196
ègetelementptr8B|
z
	full_textm
k
i%203 = getelementptr inbounds [100 x float], [100 x float]* @kernel_gpu_opencl.qB_shared, i64 0, i64 %175
&i648B

	full_text


i64 %175
Oload8BE
C
	full_text6
4
2%204 = load float, float* %203, align 4, !tbaa !25
,float*8B

	full_text

float* %203
icall8B_
]
	full_textP
N
L%205 = tail call float @llvm.fmuladd.f32(float %204, float %195, float %174)
*float8B

	full_text


float %204
*float8B

	full_text


float %195
*float8B

	full_text


float %174
Ostore8BD
B
	full_text5
3
1store float %205, float* %162, align 4, !tbaa !26
*float8B

	full_text


float %205
,float*8B

	full_text

float* %162
icall8B_
]
	full_textP
N
L%206 = tail call float @llvm.fmuladd.f32(float %204, float %198, float %173)
*float8B

	full_text


float %204
*float8B

	full_text


float %198
*float8B

	full_text


float %173
Ostore8BD
B
	full_text5
3
1store float %206, float* %163, align 4, !tbaa !28
*float8B

	full_text


float %206
,float*8B

	full_text

float* %163
icall8B_
]
	full_textP
N
L%207 = tail call float @llvm.fmuladd.f32(float %204, float %200, float %172)
*float8B

	full_text


float %204
*float8B

	full_text


float %200
*float8B

	full_text


float %172
Ostore8BD
B
	full_text5
3
1store float %207, float* %164, align 4, !tbaa !29
*float8B

	full_text


float %207
,float*8B

	full_text

float* %164
icall8B_
]
	full_textP
N
L%208 = tail call float @llvm.fmuladd.f32(float %204, float %202, float %171)
*float8B

	full_text


float %204
*float8B

	full_text


float %202
*float8B

	full_text


float %171
Ostore8BD
B
	full_text5
3
1store float %208, float* %165, align 4, !tbaa !30
*float8B

	full_text


float %208
,float*8B

	full_text

float* %165
:add8B1
/
	full_text"
 
%209 = add nuw nsw i64 %175, 1
&i648B

	full_text


i64 %175
9icmp8B/
-
	full_text 

%210 = icmp eq i64 %209, 100
&i648B

	full_text


i64 %209
=br8B5
3
	full_text&
$
"br i1 %210, label %211, label %170
$i18B

	full_text
	
i1 %210
8add8B/
-
	full_text 

%212 = add nsw i64 %156, 128
&i648B

	full_text


i64 %156
:icmp8B0
.
	full_text!

%213 = icmp slt i64 %156, -28
&i648B

	full_text


i64 %156
=br8B5
3
	full_text&
$
"br i1 %213, label %155, label %214
$i18B

	full_text
	
i1 %213
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #6
6add8B-
+
	full_text

%215 = add nuw i64 %103, 1
&i648B

	full_text


i64 %103
Jload8B@
>
	full_text1
/
-%216 = load i32, i32* %81, align 8, !tbaa !22
'i32*8B

	full_text


i32* %81
8sext8B.
,
	full_text

%217 = sext i32 %216 to i64
&i328B

	full_text


i32 %216
;icmp8B1
/
	full_text"
 
%218 = icmp slt i64 %103, %217
&i648B

	full_text


i64 %103
&i648B

	full_text


i64 %217
=br8B5
3
	full_text&
$
"br i1 %218, label %102, label %219
$i18B

	full_text
	
i1 %218
$ret8B

	full_text


ret void
5struct*8B&
$
	full_text

%struct.box_str* %2
9struct*8B*
(
	full_text

%struct.FOUR_VECTOR* %3
*float*8B

	full_text

	float* %4
9struct*8B*
(
	full_text

%struct.FOUR_VECTOR* %5
5struct*8B&
$
	full_text

%struct.par_str* %0
5struct*8B&
$
	full_text

%struct.dim_str* %1
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
2float8B%
#
	full_text

float 2.000000e+00
#i648B

	full_text	

i64 0
û[100 x %struct.FOUR_VECTOR]*8Bz
x
	full_textk
i
g@kernel_gpu_opencl.rA_shared = internal unnamed_addr global [100 x %struct.FOUR_VECTOR] undef, align 16
#i328B

	full_text	

i32 5
%i648B

	full_text
	
i64 512
%i18B

	full_text


i1 false
Ç[100 x float]*8Bl
j
	full_text]
[
Y@kernel_gpu_opencl.qB_shared = internal unnamed_addr global [100 x float] undef, align 16
%i648B

	full_text
	
i64 384
#i648B

	full_text	

i64 7
$i648B

	full_text


i64 16
%i648B

	full_text
	
i64 256
#i328B

	full_text	

i32 2
#i648B

	full_text	

i64 3
&i648B

	full_text


i64 -156
%i648B

	full_text
	
i64 -28
$i648B

	full_text


i64 32
%i648B

	full_text
	
i64 128
#i328B

	full_text	

i32 3
%i648B

	full_text
	
i64 127
#i328B

	full_text	

i32 4
$i648B

	full_text


i64 -1
&i648B

	full_text


i64 -412
3float8B&
$
	full_text

float -0.000000e+00
û[100 x %struct.FOUR_VECTOR]*8Bz
x
	full_textk
i
g@kernel_gpu_opencl.rB_shared = internal unnamed_addr global [100 x %struct.FOUR_VECTOR] undef, align 16
#i648B

	full_text	

i64 1
#i328B

	full_text	

i32 6
#i328B

	full_text	

i32 1
%i648B

	full_text
	
i64 100
%i328B

	full_text
	
i32 100
#i328B

	full_text	

i32 0        		 
 

                      !" !$ ## %& %% '( '' )* )) +, ++ -. -/ -- 01 00 23 24 22 56 55 78 77 9: 99 ;< ;; => =A @B @@ CD CE CC FG FF HI HJ HH KL KK MN MM OP OO QR QS QQ TU TT VW VV XY XX Z[ Z] \^ \\ _` __ ab ae df dd gh gg ij ik ii lm ll no nn pq pp rs rt rr uv uu wx ww yz y{ yy |} || ~ ~~ ÄÅ ÄÄ ÇÉ Ç
Ñ ÇÇ ÖÜ ÖÖ á
à áá âä â
ã ââ å
ç åå éè éé êë êê íì í
î íí ïñ ïï ó
ò óó ôö ô
õ ôô ú
ù úú ûü ûû †° †† ¢£ ¢
§ ¢¢ •¶ •• ß® ßß ©™ ©´ ¨
≠ ¨¨ ÆØ ÆÆ ∞± ∞∞ ≤≥ ≤µ ¥¥ ∂∑ ∂∂ ∏π ∏∏ ∫ª ∫∫ ºΩ ºº æø æ
¿ ææ ¡¬ ¡¡ √ƒ √
≈ √√ ∆« ∆∆ »… »»  À    Ã
Õ ÃÃ Œœ ŒŒ –
— –– “” ““ ‘’ ‘‘ ÷◊ ÷÷ ÿ⁄ ŸŸ €‹ €€ ›ﬁ ›‡ ﬂﬂ ·
‚ ·
„ ·· ‰Â ‰‰ ÊË Á
È ÁÁ ÍÎ ÍÌ ÏÏ Ó
Ô ÓÓ Ò  ÚÛ ÚÚ Ùı ÙÙ ˆ˜ ˆ˘ ¯
˙ ¯¯ ˚
¸ ˚˚ ˝˛ ˝˝ ˇÄ ˇ
Å ˇˇ Ç
É ÇÇ ÑÖ ÑÑ Üá ÜÜ àâ à
ä àà ãç å
é åå èê èì í
î íí ï
ñ ïï óò ó
ô óó ö
õ öö úù úú ûü ûû †° †
¢ †† £
§ ££ •¶ •• ß® ßß ©
™ ©© ´¨ ´´ ≠Æ ≠
Ø ≠≠ ∞± ∞∞ ≤
≥ ≤≤ ¥µ ¥
∂ ¥¥ ∑
∏ ∑∑ π∫ ππ ªº ªª Ωæ Ω
ø ΩΩ ¿
¡ ¿¿ ¬√ ¬¬ ƒ≈ ƒƒ ∆
« ∆∆ »… »»  À  
Ã    ÕŒ ÕÕ œ– œœ —“ —
‘ ”
’ ”” ÷÷ ◊ÿ ◊€ ⁄
‹ ⁄⁄ ›
ﬁ ›› ﬂ
‡ ﬂﬂ ·
‚ ·· „
‰ „„ ÂÊ Â
Á ÂÂ Ë
È ËË Í
Î ÍÍ Ï
Ì ÏÏ Ó
Ô ÓÓ Ò  ÚÛ ÚÚ Ùı ÙÙ ˆ˜ ˆˆ ¯˙ ˘
˚ ˘˘ ¸˝ ¸
˛ ¸¸ ˇÄ ˇ
Å ˇˇ ÇÉ Ç
Ñ ÇÇ Ö
Ü ÖÖ áà áá â
ä ââ ãå ãã çé ç
è çç êë êê í
ì íí îï îî ñó ññ ò
ô òò öõ öö úù ú
û úú ü† ü
° ü
¢ üü £§ ££ •
¶ •• ß® ßß ©™ ©
´ ©
¨ ©© ≠Æ ≠
Ø ≠≠ ∞± ∞
≤ ∞∞ ≥
¥ ≥≥ µ∂ µµ ∑∏ ∑∑ π∫ π
ª ππ ºΩ º
æ ºº ø¿ ø
¡ øø ¬√ ¬
ƒ ¬¬ ≈∆ ≈
« ≈≈ »… »
  »» À
Ã ÀÀ ÕŒ ÕÕ œ– œ
— œ
“ œœ ”‘ ”
’ ”” ÷◊ ÷
ÿ ÷
Ÿ ÷÷ ⁄€ ⁄
‹ ⁄⁄ ›ﬁ ›
ﬂ ›
‡ ›› ·‚ ·
„ ·· ‰Â ‰
Ê ‰
Á ‰‰ ËÈ Ë
Í ËË ÎÏ ÎÎ ÌÓ ÌÌ Ô ÔÚ ÒÒ ÛÙ ÛÛ ıˆ ı˜ ¯˘ ¯¯ ˙˚ ˙˙ ¸˝ ¸¸ ˛ˇ ˛
Ä ˛˛ ÅÇ ÅÑ Ñ ¨Ñ ·Ñ ÓÖ KÖ lÖ |Ö åÖ úÖ ˚Ö öÖ ∑Ü ÇÜ £Ü ¿á Ëá Íá Ïá Óà â 	   	  
            " $# & (' *% ,+ .% /- 10 3% 42 65 87 :9 <; >% AT B9 DV E@ G@ I) JH LF NK PM RO S@ UC WV YX [% ]T ^2 `_ b\ e• fd hd j) ki mg ol qn sp td vu xu z) {y }w | Å~ ÉÄ Ñd ÜÖ àÖ ä) ãâ çá èå ëé ìê îd ñï òï ö) õô ùó üú °û £† §d ¶d ®ß ™ ≠¨ ØÆ ±∞ ≥ µ¥ ∑ π∏ ª∂ Ωº ø∂ ¿æ ¬¡ ƒ∂ ≈√ «∆ …» À∂ ÕÃ œ∂ —– ”∂ ’∆ ◊¯ ⁄Ÿ ‹€ ﬁŸ ‡ ‚ﬂ „· Â‰ Ë È ÎÁ ÌÏ ÔÓ Ò ÛÚ ı  ˜∂ ˘Ù ˙¯ ¸˚ ˛Œ Ä˝ Å¯ ÉÇ ÖÑ áÜ â“ ä‘ ç∂ é÷ êå ìÕ îí ñí òÙ ôó õï ùö üú °û ¢ó §£ ¶• ®í ™© ¨ß Æ´ Øí ±∞ ≥∞ µÙ ∂¥ ∏≤ ∫∑ ºπ æª ø¥ ¡¿ √¬ ≈∞ «∆ …ƒ À» Ãí Œí –œ “ ‘ ’” ÿÒ €∂ ‹⁄ ﬁ⁄ ‡⁄ ‚⁄ ‰⁄ Ê∫ ÁÂ ÈÂ ÎÂ ÌÂ ÔË ÒÍ ÛÏ ıÓ ˜ˆ ˙‰ ˚Ù ˝› ˛Ú Ä÷ Å Éœ ÑÎ Ü› àÖ äâ åá éã èﬂ ëÖ ìí ï· óÖ ôò õñ ùö ûê †î °ú ¢„ §Ö ¶• ®£ ™ß ´ü ¨ç Æ© Ø ±≠ ≤∞ ¥≥ ∂µ ∏ê ∫î ªπ Ω∑ æñ ¿ö ¡ø √∑ ƒ£ ∆ß «≈ …∑  Ö ÃÀ ŒÕ –µ —Ç “œ ‘Ë ’Õ ◊º ÿˇ Ÿ÷ €Í ‹Õ ﬁ¬ ﬂ¸ ‡› ‚Ï „Õ Â» Ê˘ Á‰ ÈÓ ÍÖ ÏÎ ÓÌ ⁄ Ú⁄ ÙÛ ˆŸ ˘¨ ˚˙ ˝Ÿ ˇ¸ Ä˛ Ç  É! #! ´= \= ?≤ É≤ ¥a ´a c? @ÿ Ÿc dZ \Z @› Á› ﬂ© d© ´Í ÏÍ ”Ê Áˆ ¯ˆ å◊ Ÿ◊ ˜ã åè ”è ëŸ ⁄Å ŸÅ Éë í¯ ˘— í— ”Ô ÒÔ ˘ı ⁄ı ˜ èè ãã çç É åå ää ééΩ èè Ω† èè †Ç èè Ç÷ åå ÷µ éé µ› çç ›˜ åå ˜ü çç ü© çç ©Q èè Q´ åå ´œ çç œ÷ çç ÷¢ èè ¢‰ çç ‰ ää í èè íˇ èè ˇr èè r ãã 	ê 
ê ∑	ë 		ë 	ë ;	ë F	ë X	ë g	ë w
ë á
ë ó
ë  
ë Ã
ë –
ë ÷
ë Ÿ
ë €
ë ï
ë ©
ë ≤
ë ∆
ë ›
ë ﬂ
ë ·
ë „ë Ö
ë â
ë í
ë ò
ë •
ë Àí Fí gí wí áí óí ›í ﬂí ·í „
ì ¨
î •	ï Q	ï r
ï Ç
ï í
ï ¢
ï ˇ
ï †
ï Ωï ”ñ –ñ ©ñ ∆ñ À	ó _
ó ï	ò 5
ò ∆	ô Q	ô r
ô Ç
ô í
ô ¢
ô ˇ
ô †
ô Ω
ö Ö
ö Õ
õ ·
õ Ï
õ ò	ú 9
ù œ	û +	û -
û º
û æ
û Û	ü 	ü 	ü #	ü %	ü '	ü )
ü ¥
ü ∂
ü ∏
ü ∫
ü Ú
ü Ù	† T	† u
† ‘
† ∞
† Ò
° ·
° „
° Ó
° •	¢ 0
¢ ¡	£ 		£ 
£ Ó	§ V
§ ﬂ
• ß¶ ≥ß Ãß ïß ≤ß âß íß òß •	® 7
® »
® Î
® ¯
© ·™ ´™ ÷
™ ﬂ
™ Í
™ í™ ˜
´ Ì	¨ ≠ ≠ 	≠ 
≠ ∞
≠ ›
≠ Ë
≠ â"
kernel_gpu_opencl"
_Z12get_group_idj"
_Z12get_local_idj"
_Z7barrierj"
llvm.fmuladd.f32"	
_Z3expf"
llvm.memcpy.p0i8.p0i8.i64*†
'rodinia-3.1-lavaMD-kernel_gpu_opencl.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02Å

devmap_label


transfer_bytes
Ä‚É
 
transfer_bytes_log1p
±8tA

wgsize
Ä

wgsize_log1p
±8tA