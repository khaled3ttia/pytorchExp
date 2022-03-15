

[external]
KcallBC
A
	full_text4
2
0%9 = tail call i64 @_Z13get_global_idj(i32 1) #4
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
KcallBC
A
	full_text4
2
0%11 = tail call i64 @_Z12get_group_idj(i32 0) #4
6truncB-
+
	full_text

%12 = trunc i64 %11 to i32
#i64B

	full_text
	
i64 %11
KcallBC
A
	full_text4
2
0%13 = tail call i64 @_Z12get_local_idj(i32 0) #4
6truncB-
+
	full_text

%14 = trunc i64 %13 to i32
#i64B

	full_text
	
i64 %13
1addB*
(
	full_text

%15 = add nsw i32 %4, 1
/mulB(
&
	full_text

%16 = mul i32 %15, %5
#i32B

	full_text
	
i32 %15
0mulB)
'
	full_text

%17 = mul i32 %16, %10
#i32B

	full_text
	
i32 %16
#i32B

	full_text
	
i32 %10
4addB-
+
	full_text

%18 = add nsw i32 %17, %12
#i32B

	full_text
	
i32 %17
#i32B

	full_text
	
i32 %12
5icmpB-
+
	full_text

%19 = icmp slt i32 %14, %5
#i32B

	full_text
	
i32 %14
8brB2
0
	full_text#
!
br i1 %19, label %20, label %49
!i1B

	full_text


i1 %19
1shl8B(
&
	full_text

%21 = shl i64 %13, 32
%i648B

	full_text
	
i64 %13
9ashr8B/
-
	full_text 

%22 = ashr exact i64 %21, 32
%i648B

	full_text
	
i64 %21
6sext8B,
*
	full_text

%23 = sext i32 %15 to i64
%i328B

	full_text
	
i32 %15
6sext8B,
*
	full_text

%24 = sext i32 %18 to i64
%i328B

	full_text
	
i32 %18
5sext8B+
)
	full_text

%25 = sext i32 %5 to i64
5add8B,
*
	full_text

%26 = add nsw i64 %25, -1
%i648B

	full_text
	
i64 %25
6sub8B-
+
	full_text

%27 = sub nsw i64 %26, %22
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %22
2lshr8B(
&
	full_text

%28 = lshr i64 %27, 6
%i648B

	full_text
	
i64 %27
0and8B'
%
	full_text

%29 = and i64 %28, 1
%i648B

	full_text
	
i64 %28
5icmp8B+
)
	full_text

%30 = icmp eq i64 %29, 0
%i648B

	full_text
	
i64 %29
:br8B2
0
	full_text#
!
br i1 %30, label %31, label %45
#i18B

	full_text


i1 %30
6mul8B-
+
	full_text

%32 = mul nsw i64 %22, %23
%i648B

	full_text
	
i64 %22
%i648B

	full_text
	
i64 %23
6add8B-
+
	full_text

%33 = add nsw i64 %32, %24
%i648B

	full_text
	
i64 %32
%i648B

	full_text
	
i64 %24
rgetelementptr8B_
]
	full_textP
N
L%34 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %0, i64 %33
%i648B

	full_text
	
i64 %33
Kbitcast8B>
<
	full_text/
-
+%35 = bitcast %struct.dcomplex* %34 to i64*
-struct*8B

	full_text

struct* %34
Hload8B>
<
	full_text/
-
+%36 = load i64, i64* %35, align 8, !tbaa !8
'i64*8B

	full_text


i64* %35
êgetelementptr8B}
{
	full_textn
l
j%37 = getelementptr inbounds [64 x %struct.dcomplex], [64 x %struct.dcomplex]* @cffts2.ty1, i64 0, i64 %22
%i648B

	full_text
	
i64 %22
Kbitcast8B>
<
	full_text/
-
+%38 = bitcast %struct.dcomplex* %37 to i64*
-struct*8B

	full_text

struct* %37
Istore8B>
<
	full_text/
-
+store i64 %36, i64* %38, align 16, !tbaa !8
%i648B

	full_text
	
i64 %36
'i64*8B

	full_text


i64* %38
ygetelementptr8Bf
d
	full_textW
U
S%39 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %0, i64 %33, i32 1
%i648B

	full_text
	
i64 %33
Abitcast8B4
2
	full_text%
#
!%40 = bitcast double* %39 to i64*
-double*8B

	full_text

double* %39
Iload8B?
=
	full_text0
.
,%41 = load i64, i64* %40, align 8, !tbaa !13
'i64*8B

	full_text


i64* %40
ôgetelementptr8BÖ
Ç
	full_textu
s
q%42 = getelementptr inbounds [64 x %struct.dcomplex], [64 x %struct.dcomplex]* @cffts2.ty1, i64 0, i64 %22, i32 1
%i648B

	full_text
	
i64 %22
Abitcast8B4
2
	full_text%
#
!%43 = bitcast double* %42 to i64*
-double*8B

	full_text

double* %42
Istore8B>
<
	full_text/
-
+store i64 %41, i64* %43, align 8, !tbaa !13
%i648B

	full_text
	
i64 %41
'i64*8B

	full_text


i64* %43
5add8B,
*
	full_text

%44 = add nsw i64 %22, 64
%i648B

	full_text
	
i64 %22
'br8B

	full_text

br label %45
Dphi8B;
9
	full_text,
*
(%46 = phi i64 [ %22, %20 ], [ %44, %31 ]
%i648B

	full_text
	
i64 %22
%i648B

	full_text
	
i64 %44
5icmp8B+
)
	full_text

%47 = icmp eq i64 %28, 0
%i648B

	full_text
	
i64 %28
:br8B2
0
	full_text#
!
br i1 %47, label %49, label %48
#i18B

	full_text


i1 %47
'br8B

	full_text

br label %83
Rphi8BI
G
	full_text:
8
6%50 = phi i1 [ false, %8 ], [ %19, %83 ], [ %19, %45 ]
#i18B

	full_text


i1 %19
#i18B

	full_text


i1 %19
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
ÿcall8BÕ
 
	full_textº
π
∂tail call void @cfftz(i32 %3, i32 %7, i32 %5, %struct.dcomplex* %2, %struct.dcomplex* getelementptr inbounds ([64 x %struct.dcomplex], [64 x %struct.dcomplex]* @cffts2.ty1, i64 0, i64 0), %struct.dcomplex* getelementptr inbounds ([64 x %struct.dcomplex], [64 x %struct.dcomplex]* @cffts2.ty2, i64 0, i64 0)) #6
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
;br8B3
1
	full_text$
"
 br i1 %50, label %51, label %112
#i18B

	full_text


i1 %50
1shl8B(
&
	full_text

%52 = shl i64 %13, 32
%i648B

	full_text
	
i64 %13
9ashr8B/
-
	full_text 

%53 = ashr exact i64 %52, 32
%i648B

	full_text
	
i64 %52
6sext8B,
*
	full_text

%54 = sext i32 %15 to i64
%i328B

	full_text
	
i32 %15
6sext8B,
*
	full_text

%55 = sext i32 %18 to i64
%i328B

	full_text
	
i32 %18
5sext8B+
)
	full_text

%56 = sext i32 %5 to i64
5add8B,
*
	full_text

%57 = add nsw i64 %53, 64
%i648B

	full_text
	
i64 %53
8icmp8B.
,
	full_text

%58 = icmp sgt i64 %57, %56
%i648B

	full_text
	
i64 %57
%i648B

	full_text
	
i64 %56
Dselect8B8
6
	full_text)
'
%%59 = select i1 %58, i64 %57, i64 %56
#i18B

	full_text


i1 %58
%i648B

	full_text
	
i64 %57
%i648B

	full_text
	
i64 %56
5add8B,
*
	full_text

%60 = add nsw i64 %59, -1
%i648B

	full_text
	
i64 %59
6sub8B-
+
	full_text

%61 = sub nsw i64 %60, %53
%i648B

	full_text
	
i64 %60
%i648B

	full_text
	
i64 %53
2lshr8B(
&
	full_text

%62 = lshr i64 %61, 6
%i648B

	full_text
	
i64 %61
0and8B'
%
	full_text

%63 = and i64 %62, 1
%i648B

	full_text
	
i64 %62
5icmp8B+
)
	full_text

%64 = icmp eq i64 %63, 0
%i648B

	full_text
	
i64 %63
:br8B2
0
	full_text#
!
br i1 %64, label %65, label %79
#i18B

	full_text


i1 %64
6mul8B-
+
	full_text

%66 = mul nsw i64 %53, %54
%i648B

	full_text
	
i64 %53
%i648B

	full_text
	
i64 %54
6add8B-
+
	full_text

%67 = add nsw i64 %66, %55
%i648B

	full_text
	
i64 %66
%i648B

	full_text
	
i64 %55
êgetelementptr8B}
{
	full_textn
l
j%68 = getelementptr inbounds [64 x %struct.dcomplex], [64 x %struct.dcomplex]* @cffts2.ty1, i64 0, i64 %53
%i648B

	full_text
	
i64 %53
Kbitcast8B>
<
	full_text/
-
+%69 = bitcast %struct.dcomplex* %68 to i64*
-struct*8B

	full_text

struct* %68
Iload8B?
=
	full_text0
.
,%70 = load i64, i64* %69, align 16, !tbaa !8
'i64*8B

	full_text


i64* %69
rgetelementptr8B_
]
	full_textP
N
L%71 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %1, i64 %67
%i648B

	full_text
	
i64 %67
Kbitcast8B>
<
	full_text/
-
+%72 = bitcast %struct.dcomplex* %71 to i64*
-struct*8B

	full_text

struct* %71
Hstore8B=
;
	full_text.
,
*store i64 %70, i64* %72, align 8, !tbaa !8
%i648B

	full_text
	
i64 %70
'i64*8B

	full_text


i64* %72
ôgetelementptr8BÖ
Ç
	full_textu
s
q%73 = getelementptr inbounds [64 x %struct.dcomplex], [64 x %struct.dcomplex]* @cffts2.ty1, i64 0, i64 %53, i32 1
%i648B

	full_text
	
i64 %53
Abitcast8B4
2
	full_text%
#
!%74 = bitcast double* %73 to i64*
-double*8B

	full_text

double* %73
Iload8B?
=
	full_text0
.
,%75 = load i64, i64* %74, align 8, !tbaa !13
'i64*8B

	full_text


i64* %74
ygetelementptr8Bf
d
	full_textW
U
S%76 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %1, i64 %67, i32 1
%i648B

	full_text
	
i64 %67
Abitcast8B4
2
	full_text%
#
!%77 = bitcast double* %76 to i64*
-double*8B

	full_text

double* %76
Istore8B>
<
	full_text/
-
+store i64 %75, i64* %77, align 8, !tbaa !13
%i648B

	full_text
	
i64 %75
'i64*8B

	full_text


i64* %77
5add8B,
*
	full_text

%78 = add nsw i64 %53, 64
%i648B

	full_text
	
i64 %53
'br8B

	full_text

br label %79
Dphi8B;
9
	full_text,
*
(%80 = phi i64 [ %53, %51 ], [ %78, %65 ]
%i648B

	full_text
	
i64 %53
%i648B

	full_text
	
i64 %78
5icmp8B+
)
	full_text

%81 = icmp eq i64 %62, 0
%i648B

	full_text
	
i64 %62
;br8B3
1
	full_text$
"
 br i1 %81, label %112, label %82
#i18B

	full_text


i1 %81
(br8	B 

	full_text

br label %113
Ephi8
B<
:
	full_text-
+
)%84 = phi i64 [ %46, %48 ], [ %110, %83 ]
%i648
B

	full_text
	
i64 %46
&i648
B

	full_text


i64 %110
6mul8
B-
+
	full_text

%85 = mul nsw i64 %84, %23
%i648
B

	full_text
	
i64 %84
%i648
B

	full_text
	
i64 %23
6add8
B-
+
	full_text

%86 = add nsw i64 %85, %24
%i648
B

	full_text
	
i64 %85
%i648
B

	full_text
	
i64 %24
rgetelementptr8
B_
]
	full_textP
N
L%87 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %0, i64 %86
%i648
B

	full_text
	
i64 %86
Kbitcast8
B>
<
	full_text/
-
+%88 = bitcast %struct.dcomplex* %87 to i64*
-struct*8
B

	full_text

struct* %87
Hload8
B>
<
	full_text/
-
+%89 = load i64, i64* %88, align 8, !tbaa !8
'i64*8
B

	full_text


i64* %88
êgetelementptr8
B}
{
	full_textn
l
j%90 = getelementptr inbounds [64 x %struct.dcomplex], [64 x %struct.dcomplex]* @cffts2.ty1, i64 0, i64 %84
%i648
B

	full_text
	
i64 %84
Kbitcast8
B>
<
	full_text/
-
+%91 = bitcast %struct.dcomplex* %90 to i64*
-struct*8
B

	full_text

struct* %90
Istore8
B>
<
	full_text/
-
+store i64 %89, i64* %91, align 16, !tbaa !8
%i648
B

	full_text
	
i64 %89
'i64*8
B

	full_text


i64* %91
ygetelementptr8
Bf
d
	full_textW
U
S%92 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %0, i64 %86, i32 1
%i648
B

	full_text
	
i64 %86
Abitcast8
B4
2
	full_text%
#
!%93 = bitcast double* %92 to i64*
-double*8
B

	full_text

double* %92
Iload8
B?
=
	full_text0
.
,%94 = load i64, i64* %93, align 8, !tbaa !13
'i64*8
B

	full_text


i64* %93
ôgetelementptr8
BÖ
Ç
	full_textu
s
q%95 = getelementptr inbounds [64 x %struct.dcomplex], [64 x %struct.dcomplex]* @cffts2.ty1, i64 0, i64 %84, i32 1
%i648
B

	full_text
	
i64 %84
Abitcast8
B4
2
	full_text%
#
!%96 = bitcast double* %95 to i64*
-double*8
B

	full_text

double* %95
Istore8
B>
<
	full_text/
-
+store i64 %94, i64* %96, align 8, !tbaa !13
%i648
B

	full_text
	
i64 %94
'i64*8
B

	full_text


i64* %96
5add8
B,
*
	full_text

%97 = add nsw i64 %84, 64
%i648
B

	full_text
	
i64 %84
6mul8
B-
+
	full_text

%98 = mul nsw i64 %97, %23
%i648
B

	full_text
	
i64 %97
%i648
B

	full_text
	
i64 %23
6add8
B-
+
	full_text

%99 = add nsw i64 %98, %24
%i648
B

	full_text
	
i64 %98
%i648
B

	full_text
	
i64 %24
sgetelementptr8
B`
^
	full_textQ
O
M%100 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %0, i64 %99
%i648
B

	full_text
	
i64 %99
Mbitcast8
B@
>
	full_text1
/
-%101 = bitcast %struct.dcomplex* %100 to i64*
.struct*8
B

	full_text

struct* %100
Jload8
B@
>
	full_text1
/
-%102 = load i64, i64* %101, align 8, !tbaa !8
(i64*8
B

	full_text

	i64* %101
ëgetelementptr8
B~
|
	full_texto
m
k%103 = getelementptr inbounds [64 x %struct.dcomplex], [64 x %struct.dcomplex]* @cffts2.ty1, i64 0, i64 %97
%i648
B

	full_text
	
i64 %97
Mbitcast8
B@
>
	full_text1
/
-%104 = bitcast %struct.dcomplex* %103 to i64*
.struct*8
B

	full_text

struct* %103
Kstore8
B@
>
	full_text1
/
-store i64 %102, i64* %104, align 16, !tbaa !8
&i648
B

	full_text


i64 %102
(i64*8
B

	full_text

	i64* %104
zgetelementptr8
Bg
e
	full_textX
V
T%105 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %0, i64 %99, i32 1
%i648
B

	full_text
	
i64 %99
Cbitcast8
B6
4
	full_text'
%
#%106 = bitcast double* %105 to i64*
.double*8
B

	full_text

double* %105
Kload8
BA
?
	full_text2
0
.%107 = load i64, i64* %106, align 8, !tbaa !13
(i64*8
B

	full_text

	i64* %106
ögetelementptr8
BÜ
É
	full_textv
t
r%108 = getelementptr inbounds [64 x %struct.dcomplex], [64 x %struct.dcomplex]* @cffts2.ty1, i64 0, i64 %97, i32 1
%i648
B

	full_text
	
i64 %97
Cbitcast8
B6
4
	full_text'
%
#%109 = bitcast double* %108 to i64*
.double*8
B

	full_text

double* %108
Kstore8
B@
>
	full_text1
/
-store i64 %107, i64* %109, align 8, !tbaa !13
&i648
B

	full_text


i64 %107
(i64*8
B

	full_text

	i64* %109
7add8
B.
,
	full_text

%110 = add nsw i64 %84, 128
%i648
B

	full_text
	
i64 %84
:icmp8
B0
.
	full_text!

%111 = icmp slt i64 %110, %25
&i648
B

	full_text


i64 %110
%i648
B

	full_text
	
i64 %25
;br8
B3
1
	full_text$
"
 br i1 %111, label %83, label %49
$i18
B

	full_text
	
i1 %111
$ret8B

	full_text


ret void
Gphi8B>
<
	full_text/
-
+%114 = phi i64 [ %80, %82 ], [ %140, %113 ]
%i648B

	full_text
	
i64 %80
&i648B

	full_text


i64 %140
8mul8B/
-
	full_text 

%115 = mul nsw i64 %114, %54
&i648B

	full_text


i64 %114
%i648B

	full_text
	
i64 %54
8add8B/
-
	full_text 

%116 = add nsw i64 %115, %55
&i648B

	full_text


i64 %115
%i648B

	full_text
	
i64 %55
ígetelementptr8B
}
	full_textp
n
l%117 = getelementptr inbounds [64 x %struct.dcomplex], [64 x %struct.dcomplex]* @cffts2.ty1, i64 0, i64 %114
&i648B

	full_text


i64 %114
Mbitcast8B@
>
	full_text1
/
-%118 = bitcast %struct.dcomplex* %117 to i64*
.struct*8B

	full_text

struct* %117
Kload8BA
?
	full_text2
0
.%119 = load i64, i64* %118, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %118
tgetelementptr8Ba
_
	full_textR
P
N%120 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %1, i64 %116
&i648B

	full_text


i64 %116
Mbitcast8B@
>
	full_text1
/
-%121 = bitcast %struct.dcomplex* %120 to i64*
.struct*8B

	full_text

struct* %120
Jstore8B?
=
	full_text0
.
,store i64 %119, i64* %121, align 8, !tbaa !8
&i648B

	full_text


i64 %119
(i64*8B

	full_text

	i64* %121
õgetelementptr8Bá
Ñ
	full_textw
u
s%122 = getelementptr inbounds [64 x %struct.dcomplex], [64 x %struct.dcomplex]* @cffts2.ty1, i64 0, i64 %114, i32 1
&i648B

	full_text


i64 %114
Cbitcast8B6
4
	full_text'
%
#%123 = bitcast double* %122 to i64*
.double*8B

	full_text

double* %122
Kload8BA
?
	full_text2
0
.%124 = load i64, i64* %123, align 8, !tbaa !13
(i64*8B

	full_text

	i64* %123
{getelementptr8Bh
f
	full_textY
W
U%125 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %1, i64 %116, i32 1
&i648B

	full_text


i64 %116
Cbitcast8B6
4
	full_text'
%
#%126 = bitcast double* %125 to i64*
.double*8B

	full_text

double* %125
Kstore8B@
>
	full_text1
/
-store i64 %124, i64* %126, align 8, !tbaa !13
&i648B

	full_text


i64 %124
(i64*8B

	full_text

	i64* %126
7add8B.
,
	full_text

%127 = add nsw i64 %114, 64
&i648B

	full_text


i64 %114
8mul8B/
-
	full_text 

%128 = mul nsw i64 %127, %54
&i648B

	full_text


i64 %127
%i648B

	full_text
	
i64 %54
8add8B/
-
	full_text 

%129 = add nsw i64 %128, %55
&i648B

	full_text


i64 %128
%i648B

	full_text
	
i64 %55
ígetelementptr8B
}
	full_textp
n
l%130 = getelementptr inbounds [64 x %struct.dcomplex], [64 x %struct.dcomplex]* @cffts2.ty1, i64 0, i64 %127
&i648B

	full_text


i64 %127
Mbitcast8B@
>
	full_text1
/
-%131 = bitcast %struct.dcomplex* %130 to i64*
.struct*8B

	full_text

struct* %130
Kload8BA
?
	full_text2
0
.%132 = load i64, i64* %131, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %131
tgetelementptr8Ba
_
	full_textR
P
N%133 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %1, i64 %129
&i648B

	full_text


i64 %129
Mbitcast8B@
>
	full_text1
/
-%134 = bitcast %struct.dcomplex* %133 to i64*
.struct*8B

	full_text

struct* %133
Jstore8B?
=
	full_text0
.
,store i64 %132, i64* %134, align 8, !tbaa !8
&i648B

	full_text


i64 %132
(i64*8B

	full_text

	i64* %134
õgetelementptr8Bá
Ñ
	full_textw
u
s%135 = getelementptr inbounds [64 x %struct.dcomplex], [64 x %struct.dcomplex]* @cffts2.ty1, i64 0, i64 %127, i32 1
&i648B

	full_text


i64 %127
Cbitcast8B6
4
	full_text'
%
#%136 = bitcast double* %135 to i64*
.double*8B

	full_text

double* %135
Kload8BA
?
	full_text2
0
.%137 = load i64, i64* %136, align 8, !tbaa !13
(i64*8B

	full_text

	i64* %136
{getelementptr8Bh
f
	full_textY
W
U%138 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %1, i64 %129, i32 1
&i648B

	full_text


i64 %129
Cbitcast8B6
4
	full_text'
%
#%139 = bitcast double* %138 to i64*
.double*8B

	full_text

double* %138
Kstore8B@
>
	full_text1
/
-store i64 %137, i64* %139, align 8, !tbaa !13
&i648B

	full_text


i64 %137
(i64*8B

	full_text

	i64* %139
8add8B/
-
	full_text 

%140 = add nsw i64 %114, 128
&i648B

	full_text


i64 %114
:icmp8B0
.
	full_text!

%141 = icmp slt i64 %140, %56
&i648B

	full_text


i64 %140
%i648B

	full_text
	
i64 %56
=br8B5
3
	full_text&
$
"br i1 %141, label %113, label %112
$i18B

	full_text
	
i1 %141
6struct*8B'
%
	full_text

%struct.dcomplex* %2
$i328B

	full_text


i32 %3
$i328B

	full_text


i32 %7
$i328B

	full_text


i32 %4
6struct*8B'
%
	full_text

%struct.dcomplex* %1
$i328B

	full_text


i32 %5
6struct*8B'
%
	full_text

%struct.dcomplex* %0
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
%i18B

	full_text


i1 false
östruct*8Bä
á
	full_textz
x
v%struct.dcomplex* getelementptr inbounds ([64 x %struct.dcomplex], [64 x %struct.dcomplex]* @cffts2.ty1, i64 0, i64 0)
östruct*8Bä
á
	full_textz
x
v%struct.dcomplex* getelementptr inbounds ([64 x %struct.dcomplex], [64 x %struct.dcomplex]* @cffts2.ty2, i64 0, i64 0)
#i648B

	full_text	

i64 6
#i328B

	full_text	

i32 1
#i328B

	full_text	

i32 0
$i648B

	full_text


i64 -1
#i648B

	full_text	

i64 1
#i648B

	full_text	

i64 0
x[64 x %struct.dcomplex]*8BX
V
	full_textI
G
E@cffts2.ty1 = internal global [64 x %struct.dcomplex] undef, align 16
$i648B

	full_text


i64 64
%i648B

	full_text
	
i64 128
$i648B

	full_text


i64 32       	  

                      !    "# "$ "" %& %% '( '' )* )) +, +. -/ -- 01 02 00 34 33 56 55 78 77 9: 99 ;< ;; => =? == @A @@ BC BB DE DD FG FF HI HH JK JL JJ MN MM OQ PR PP ST SS UV UY XZ XX [[ \\ ]] ^_ ^a `` bc bb de dd fg ff hh ij ii kl km kk no np nq nn rs rr tu tv tt wx ww yz yy {| {{ }~ }Ä 	Å  ÇÉ Ç
Ñ ÇÇ Ö
Ü ÖÖ áà áá âä ââ ã
å ãã çé çç èê è
ë èè í
ì íí îï îî ñó ññ ò
ô òò öõ öö úù ú
û úú ü† üü °£ ¢
§ ¢¢ •¶ •• ß® ß´ ™
¨ ™™ ≠Æ ≠
Ø ≠≠ ∞± ∞
≤ ∞∞ ≥
¥ ≥≥ µ∂ µµ ∑∏ ∑∑ π
∫ ππ ªº ªª Ωæ Ω
ø ΩΩ ¿
¡ ¿¿ ¬√ ¬¬ ƒ≈ ƒƒ ∆
« ∆∆ »… »»  À  
Ã    ÕŒ ÕÕ œ– œ
— œœ “” “
‘ ““ ’
÷ ’’ ◊ÿ ◊◊ Ÿ⁄ ŸŸ €
‹ €€ ›ﬁ ›› ﬂ‡ ﬂ
· ﬂﬂ ‚
„ ‚‚ ‰Â ‰‰ ÊÁ ÊÊ Ë
È ËË ÍÎ ÍÍ ÏÌ Ï
Ó ÏÏ Ô ÔÔ ÒÚ Ò
Û ÒÒ Ùı Ù¯ ˜
˘ ˜˜ ˙˚ ˙
¸ ˙˙ ˝˛ ˝
ˇ ˝˝ Ä
Å ÄÄ ÇÉ ÇÇ ÑÖ ÑÑ Ü
á ÜÜ àâ àà äã ä
å ää ç
é çç èê èè ëí ëë ì
î ìì ïñ ïï óò ó
ô óó öõ öö úù ú
û úú ü† ü
° üü ¢
£ ¢¢ §• §§ ¶ß ¶¶ ®
© ®® ™´ ™™ ¨≠ ¨
Æ ¨¨ Ø
∞ ØØ ±≤ ±± ≥¥ ≥≥ µ
∂ µµ ∑∏ ∑∑ π∫ π
ª ππ ºΩ ºº æø æ
¿ ææ ¡¬ ¡	√ \ƒ \	≈ \∆ 
« ã« ò« Ü« ì« ®« µ	» 	» » 	» \» h… 3… @… ≥… ¿… ’… ‚   	
         
   !  # $" &% (' *) , . /- 1 20 43 65 8 :9 <7 >; ?0 A@ CB E GF ID KH L N QM R% TS V Y ZX _ a` c
 e gb ji lh mk oi ph qn sr ub vt xw zy |{ ~b Äd Å Éf Ñb ÜÖ àá äÇ åã éâ êç ëb ìí ïî óÇ ôò õñ ùö ûb †b £ü §w ¶• ®P ´Ô ¨™ Æ Ø≠ ± ≤∞ ¥≥ ∂µ ∏™ ∫π º∑ æª ø∞ ¡¿ √¬ ≈™ «∆ …ƒ À» Ã™ ŒÕ – —œ ” ‘“ ÷’ ÿ◊ ⁄Õ ‹€ ﬁŸ ‡› ·“ „‚ Â‰ ÁÕ ÈË ÎÊ ÌÍ Ó™ Ô Ú ÛÒ ı¢ ¯º ˘˜ ˚d ¸˙ ˛f ˇ˜ ÅÄ ÉÇ Ö˝ áÜ âÑ ãà å˜ éç êè í˝ îì ñë òï ô˜ õö ùd ûú †f °ö £¢ •§ ßü ©® ´¶ ≠™ Æö ∞Ø ≤± ¥ü ∂µ ∏≥ ∫∑ ª˜ Ωº øh ¿æ ¬  X+ -+ P^ `^ ˆO PU XU W} } ¢W ™° ¢ß ˆß ©Ù ™Ù X© ˜¡ ˜¡ ˆ ˆ ŒŒ    ÃÃ ÀÀ ÕÕ    [ ÕÕ [\ ŒŒ \] ÕÕ ] ÀÀ  ÃÃ œ X	– \	— \	“ %	“ w” 	” 
	” @	” F” [” ]
” í
” ò
” ¿
” ∆
” ‚
” Ë
” ç
” ì
” Ø
” µ‘ ‘ 	’  	’ r	÷ '	÷ y	◊ )	◊ 9	◊ F	◊ S	◊ {
◊ Ö
◊ í
◊ •
◊ π
◊ ∆
◊ €
◊ Ë
◊ Ä
◊ ç
◊ ¢
◊ Øÿ 9ÿ Fÿ Öÿ íÿ πÿ ∆ÿ €ÿ Ëÿ Äÿ çÿ ¢ÿ Ø	Ÿ M	Ÿ i
Ÿ ü
Ÿ Õ
Ÿ ö
⁄ Ô
⁄ º	€ 	€ 	€ `	€ b"
cffts2"
_Z13get_global_idj"
_Z12get_group_idj"
_Z12get_local_idj"
_Z7barrierj"
cfftz*â
npb-FT-cffts2.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02Ä

wgsize_log1p
@sÅA
 
transfer_bytes_log1p
@sÅA

devmap_label


wgsize
@

transfer_bytes
êîä