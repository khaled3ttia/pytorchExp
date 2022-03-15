

[external]
JcallBB
@
	full_text3
1
/%7 = tail call i64 @_Z12get_group_idj(i32 0) #4
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
/%9 = tail call i64 @_Z12get_local_idj(i32 0) #4
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
5icmpB-
+
	full_text

%11 = icmp slt i32 %10, 64
#i32B

	full_text
	
i32 %10
8brB2
0
	full_text#
!
br i1 %11, label %12, label %75
!i1B

	full_text


i1 %11
4mul8B+
)
	full_text

%13 = mul nsw i32 %5, %4
5add8B,
*
	full_text

%14 = add nsw i32 %13, %5
%i328B

	full_text
	
i32 %13
0shl8B'
%
	full_text

%15 = shl i64 %9, 32
$i648B

	full_text


i64 %9
9ashr8B/
-
	full_text 

%16 = ashr exact i64 %15, 32
%i648B

	full_text
	
i64 %15
6sext8B,
*
	full_text

%17 = sext i32 %14 to i64
%i328B

	full_text
	
i32 %14
5sext8B+
)
	full_text

%18 = sext i32 %4 to i64
'br8B

	full_text

br label %19
Dphi8B;
9
	full_text,
*
(%20 = phi i64 [ %17, %12 ], [ %40, %19 ]
%i648B

	full_text
	
i64 %17
%i648B

	full_text
	
i64 %40
Bphi8B9
7
	full_text*
(
&%21 = phi i64 [ 0, %12 ], [ %41, %19 ]
%i648B

	full_text
	
i64 %41
6add8B-
+
	full_text

%22 = add nsw i64 %20, %16
%i648B

	full_text
	
i64 %20
%i648B

	full_text
	
i64 %16
\getelementptr8BI
G
	full_text:
8
6%23 = getelementptr inbounds float, float* %0, i64 %22
%i648B

	full_text
	
i64 %22
@bitcast8B3
1
	full_text$
"
 %24 = bitcast float* %23 to i32*
+float*8B

	full_text


float* %23
Hload8B>
<
	full_text/
-
+%25 = load i32, i32* %24, align 4, !tbaa !8
'i32*8B

	full_text


i32* %24
0shl8B'
%
	full_text

%26 = shl i64 %21, 6
%i648B

	full_text
	
i64 %21
6add8B-
+
	full_text

%27 = add nsw i64 %26, %16
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %16
\getelementptr8BI
G
	full_text:
8
6%28 = getelementptr inbounds float, float* %1, i64 %27
%i648B

	full_text
	
i64 %27
@bitcast8B3
1
	full_text$
"
 %29 = bitcast float* %28 to i32*
+float*8B

	full_text


float* %28
Hstore8B=
;
	full_text.
,
*store i32 %25, i32* %29, align 4, !tbaa !8
%i328B

	full_text
	
i32 %25
'i32*8B

	full_text


i32* %29
6add8B-
+
	full_text

%30 = add nsw i64 %20, %18
%i648B

	full_text
	
i64 %20
%i648B

	full_text
	
i64 %18
6add8B-
+
	full_text

%31 = add nsw i64 %30, %16
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %16
\getelementptr8BI
G
	full_text:
8
6%32 = getelementptr inbounds float, float* %0, i64 %31
%i648B

	full_text
	
i64 %31
@bitcast8B3
1
	full_text$
"
 %33 = bitcast float* %32 to i32*
+float*8B

	full_text


float* %32
Hload8B>
<
	full_text/
-
+%34 = load i32, i32* %33, align 4, !tbaa !8
'i32*8B

	full_text


i32* %33
0shl8B'
%
	full_text

%35 = shl i64 %21, 6
%i648B

	full_text
	
i64 %21
/or8B'
%
	full_text

%36 = or i64 %35, 64
%i648B

	full_text
	
i64 %35
6add8B-
+
	full_text

%37 = add nsw i64 %36, %16
%i648B

	full_text
	
i64 %36
%i648B

	full_text
	
i64 %16
\getelementptr8BI
G
	full_text:
8
6%38 = getelementptr inbounds float, float* %1, i64 %37
%i648B

	full_text
	
i64 %37
@bitcast8B3
1
	full_text$
"
 %39 = bitcast float* %38 to i32*
+float*8B

	full_text


float* %38
Hstore8B=
;
	full_text.
,
*store i32 %34, i32* %39, align 4, !tbaa !8
%i328B

	full_text
	
i32 %34
'i32*8B

	full_text


i32* %39
6add8B-
+
	full_text

%40 = add nsw i64 %30, %18
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %18
4add8B+
)
	full_text

%41 = add nsw i64 %21, 2
%i648B

	full_text
	
i64 %21
6icmp8B,
*
	full_text

%42 = icmp eq i64 %41, 32
%i648B

	full_text
	
i64 %41
:br8B2
0
	full_text#
!
br i1 %42, label %43, label %19
#i18B

	full_text


i1 %42
/shl8B&
$
	full_text

%44 = shl i32 %8, 6
$i328B

	full_text


i32 %8
1add8B(
&
	full_text

%45 = add i32 %44, 64
%i328B

	full_text
	
i32 %44
2add8B)
'
	full_text

%46 = add i32 %45, %10
%i328B

	full_text
	
i32 %45
%i328B

	full_text
	
i32 %10
'br8B

	full_text

br label %47
Dphi8B;
9
	full_text,
*
(%48 = phi i64 [ %17, %43 ], [ %72, %47 ]
%i648B

	full_text
	
i64 %17
%i648B

	full_text
	
i64 %72
Bphi8B9
7
	full_text*
(
&%49 = phi i64 [ 0, %43 ], [ %73, %47 ]
%i648B

	full_text
	
i64 %73
8trunc8B-
+
	full_text

%50 = trunc i64 %48 to i32
%i648B

	full_text
	
i64 %48
2add8B)
'
	full_text

%51 = add i32 %46, %50
%i328B

	full_text
	
i32 %46
%i328B

	full_text
	
i32 %50
6sext8B,
*
	full_text

%52 = sext i32 %51 to i64
%i328B

	full_text
	
i32 %51
\getelementptr8BI
G
	full_text:
8
6%53 = getelementptr inbounds float, float* %0, i64 %52
%i648B

	full_text
	
i64 %52
@bitcast8B3
1
	full_text$
"
 %54 = bitcast float* %53 to i32*
+float*8B

	full_text


float* %53
Hload8B>
<
	full_text/
-
+%55 = load i32, i32* %54, align 4, !tbaa !8
'i32*8B

	full_text


i32* %54
0shl8B'
%
	full_text

%56 = shl i64 %49, 6
%i648B

	full_text
	
i64 %49
6add8B-
+
	full_text

%57 = add nsw i64 %56, %16
%i648B

	full_text
	
i64 %56
%i648B

	full_text
	
i64 %16
\getelementptr8BI
G
	full_text:
8
6%58 = getelementptr inbounds float, float* %2, i64 %57
%i648B

	full_text
	
i64 %57
@bitcast8B3
1
	full_text$
"
 %59 = bitcast float* %58 to i32*
+float*8B

	full_text


float* %58
Hstore8B=
;
	full_text.
,
*store i32 %55, i32* %59, align 4, !tbaa !8
%i328B

	full_text
	
i32 %55
'i32*8B

	full_text


i32* %59
6add8B-
+
	full_text

%60 = add nsw i64 %48, %18
%i648B

	full_text
	
i64 %48
%i648B

	full_text
	
i64 %18
8trunc8B-
+
	full_text

%61 = trunc i64 %60 to i32
%i648B

	full_text
	
i64 %60
2add8B)
'
	full_text

%62 = add i32 %46, %61
%i328B

	full_text
	
i32 %46
%i328B

	full_text
	
i32 %61
6sext8B,
*
	full_text

%63 = sext i32 %62 to i64
%i328B

	full_text
	
i32 %62
\getelementptr8BI
G
	full_text:
8
6%64 = getelementptr inbounds float, float* %0, i64 %63
%i648B

	full_text
	
i64 %63
@bitcast8B3
1
	full_text$
"
 %65 = bitcast float* %64 to i32*
+float*8B

	full_text


float* %64
Hload8B>
<
	full_text/
-
+%66 = load i32, i32* %65, align 4, !tbaa !8
'i32*8B

	full_text


i32* %65
0shl8B'
%
	full_text

%67 = shl i64 %49, 6
%i648B

	full_text
	
i64 %49
/or8B'
%
	full_text

%68 = or i64 %67, 64
%i648B

	full_text
	
i64 %67
6add8B-
+
	full_text

%69 = add nsw i64 %68, %16
%i648B

	full_text
	
i64 %68
%i648B

	full_text
	
i64 %16
\getelementptr8BI
G
	full_text:
8
6%70 = getelementptr inbounds float, float* %2, i64 %69
%i648B

	full_text
	
i64 %69
@bitcast8B3
1
	full_text$
"
 %71 = bitcast float* %70 to i32*
+float*8B

	full_text


float* %70
Hstore8B=
;
	full_text.
,
*store i32 %66, i32* %71, align 4, !tbaa !8
%i328B

	full_text
	
i32 %66
'i32*8B

	full_text


i32* %71
6add8B-
+
	full_text

%72 = add nsw i64 %60, %18
%i648B

	full_text
	
i64 %60
%i648B

	full_text
	
i64 %18
4add8B+
)
	full_text

%73 = add nsw i64 %49, 2
%i648B

	full_text
	
i64 %49
6icmp8B,
*
	full_text

%74 = icmp eq i64 %73, 64
%i648B

	full_text
	
i64 %73
;br8B3
1
	full_text$
"
 br i1 %74, label %139, label %47
#i18B

	full_text


i1 %74
4add8B+
)
	full_text

%76 = add nsw i32 %5, 32
5mul8B,
*
	full_text

%77 = mul nsw i32 %76, %4
%i328B

	full_text
	
i32 %76
5add8B,
*
	full_text

%78 = add nsw i32 %77, %5
%i328B

	full_text
	
i32 %77
0shl8B'
%
	full_text

%79 = shl i64 %9, 32
$i648B

	full_text


i64 %9
<add8B3
1
	full_text$
"
 %80 = add i64 %79, -274877906944
%i648B

	full_text
	
i64 %79
9ashr8B/
-
	full_text 

%81 = ashr exact i64 %80, 32
%i648B

	full_text
	
i64 %80
6sext8B,
*
	full_text

%82 = sext i32 %78 to i64
%i328B

	full_text
	
i32 %78
5sext8B+
)
	full_text

%83 = sext i32 %4 to i64
'br8B

	full_text

br label %84
Ephi8B<
:
	full_text-
+
)%85 = phi i64 [ %82, %75 ], [ %105, %84 ]
%i648B

	full_text
	
i64 %82
&i648B

	full_text


i64 %105
Dphi8B;
9
	full_text,
*
(%86 = phi i64 [ 32, %75 ], [ %106, %84 ]
&i648B

	full_text


i64 %106
6add8B-
+
	full_text

%87 = add nsw i64 %85, %81
%i648B

	full_text
	
i64 %85
%i648B

	full_text
	
i64 %81
\getelementptr8BI
G
	full_text:
8
6%88 = getelementptr inbounds float, float* %0, i64 %87
%i648B

	full_text
	
i64 %87
@bitcast8B3
1
	full_text$
"
 %89 = bitcast float* %88 to i32*
+float*8B

	full_text


float* %88
Hload8B>
<
	full_text/
-
+%90 = load i32, i32* %89, align 4, !tbaa !8
'i32*8B

	full_text


i32* %89
0shl8B'
%
	full_text

%91 = shl i64 %86, 6
%i648B

	full_text
	
i64 %86
6add8B-
+
	full_text

%92 = add nsw i64 %91, %81
%i648B

	full_text
	
i64 %91
%i648B

	full_text
	
i64 %81
\getelementptr8BI
G
	full_text:
8
6%93 = getelementptr inbounds float, float* %1, i64 %92
%i648B

	full_text
	
i64 %92
@bitcast8B3
1
	full_text$
"
 %94 = bitcast float* %93 to i32*
+float*8B

	full_text


float* %93
Hstore8B=
;
	full_text.
,
*store i32 %90, i32* %94, align 4, !tbaa !8
%i328B

	full_text
	
i32 %90
'i32*8B

	full_text


i32* %94
6add8B-
+
	full_text

%95 = add nsw i64 %85, %83
%i648B

	full_text
	
i64 %85
%i648B

	full_text
	
i64 %83
6add8B-
+
	full_text

%96 = add nsw i64 %95, %81
%i648B

	full_text
	
i64 %95
%i648B

	full_text
	
i64 %81
\getelementptr8BI
G
	full_text:
8
6%97 = getelementptr inbounds float, float* %0, i64 %96
%i648B

	full_text
	
i64 %96
@bitcast8B3
1
	full_text$
"
 %98 = bitcast float* %97 to i32*
+float*8B

	full_text


float* %97
Hload8B>
<
	full_text/
-
+%99 = load i32, i32* %98, align 4, !tbaa !8
'i32*8B

	full_text


i32* %98
1shl8B(
&
	full_text

%100 = shl i64 %86, 6
%i648B

	full_text
	
i64 %86
1or8B)
'
	full_text

%101 = or i64 %100, 64
&i648B

	full_text


i64 %100
8add8B/
-
	full_text 

%102 = add nsw i64 %101, %81
&i648B

	full_text


i64 %101
%i648B

	full_text
	
i64 %81
^getelementptr8BK
I
	full_text<
:
8%103 = getelementptr inbounds float, float* %1, i64 %102
&i648B

	full_text


i64 %102
Bbitcast8B5
3
	full_text&
$
"%104 = bitcast float* %103 to i32*
,float*8B

	full_text

float* %103
Istore8B>
<
	full_text/
-
+store i32 %99, i32* %104, align 4, !tbaa !8
%i328B

	full_text
	
i32 %99
(i32*8B

	full_text

	i32* %104
7add8B.
,
	full_text

%105 = add nsw i64 %95, %83
%i648B

	full_text
	
i64 %95
%i648B

	full_text
	
i64 %83
5add8B,
*
	full_text

%106 = add nsw i64 %86, 2
%i648B

	full_text
	
i64 %86
8icmp8B.
,
	full_text

%107 = icmp eq i64 %106, 64
&i648B

	full_text


i64 %106
<br8B4
2
	full_text%
#
!br i1 %107, label %108, label %84
$i18B

	full_text
	
i1 %107
0shl8B'
%
	full_text

%109 = shl i32 %8, 6
$i328B

	full_text


i32 %8
1add8B(
&
	full_text

%110 = add i32 %5, 64
5add8B,
*
	full_text

%111 = add i32 %110, %109
&i328B

	full_text


i32 %110
&i328B

	full_text


i32 %109
7mul8B.
,
	full_text

%112 = mul nsw i32 %111, %4
&i328B

	full_text


i32 %111
7add8B.
,
	full_text

%113 = add nsw i32 %112, %5
&i328B

	full_text


i32 %112
8sext8B.
,
	full_text

%114 = sext i32 %113 to i64
&i328B

	full_text


i32 %113
(br8B 

	full_text

br label %115
Iphi8B@
>
	full_text1
/
-%116 = phi i64 [ %114, %108 ], [ %136, %115 ]
&i648B

	full_text


i64 %114
&i648B

	full_text


i64 %136
Fphi8B=
;
	full_text.
,
*%117 = phi i64 [ 0, %108 ], [ %137, %115 ]
&i648B

	full_text


i64 %137
8add8B/
-
	full_text 

%118 = add nsw i64 %116, %81
&i648B

	full_text


i64 %116
%i648B

	full_text
	
i64 %81
^getelementptr8BK
I
	full_text<
:
8%119 = getelementptr inbounds float, float* %0, i64 %118
&i648B

	full_text


i64 %118
Bbitcast8B5
3
	full_text&
$
"%120 = bitcast float* %119 to i32*
,float*8B

	full_text

float* %119
Jload8B@
>
	full_text1
/
-%121 = load i32, i32* %120, align 4, !tbaa !8
(i32*8B

	full_text

	i32* %120
2shl8B)
'
	full_text

%122 = shl i64 %117, 6
&i648B

	full_text


i64 %117
8add8B/
-
	full_text 

%123 = add nsw i64 %122, %81
&i648B

	full_text


i64 %122
%i648B

	full_text
	
i64 %81
^getelementptr8BK
I
	full_text<
:
8%124 = getelementptr inbounds float, float* %3, i64 %123
&i648B

	full_text


i64 %123
Bbitcast8B5
3
	full_text&
$
"%125 = bitcast float* %124 to i32*
,float*8B

	full_text

float* %124
Jstore8B?
=
	full_text0
.
,store i32 %121, i32* %125, align 4, !tbaa !8
&i328B

	full_text


i32 %121
(i32*8B

	full_text

	i32* %125
8add8B/
-
	full_text 

%126 = add nsw i64 %116, %83
&i648B

	full_text


i64 %116
%i648B

	full_text
	
i64 %83
8add8B/
-
	full_text 

%127 = add nsw i64 %126, %81
&i648B

	full_text


i64 %126
%i648B

	full_text
	
i64 %81
^getelementptr8BK
I
	full_text<
:
8%128 = getelementptr inbounds float, float* %0, i64 %127
&i648B

	full_text


i64 %127
Bbitcast8B5
3
	full_text&
$
"%129 = bitcast float* %128 to i32*
,float*8B

	full_text

float* %128
Jload8B@
>
	full_text1
/
-%130 = load i32, i32* %129, align 4, !tbaa !8
(i32*8B

	full_text

	i32* %129
2shl8B)
'
	full_text

%131 = shl i64 %117, 6
&i648B

	full_text


i64 %117
1or8B)
'
	full_text

%132 = or i64 %131, 64
&i648B

	full_text


i64 %131
8add8B/
-
	full_text 

%133 = add nsw i64 %132, %81
&i648B

	full_text


i64 %132
%i648B

	full_text
	
i64 %81
^getelementptr8BK
I
	full_text<
:
8%134 = getelementptr inbounds float, float* %3, i64 %133
&i648B

	full_text


i64 %133
Bbitcast8B5
3
	full_text&
$
"%135 = bitcast float* %134 to i32*
,float*8B

	full_text

float* %134
Jstore8B?
=
	full_text0
.
,store i32 %130, i32* %135, align 4, !tbaa !8
&i328B

	full_text


i32 %130
(i32*8B

	full_text

	i32* %135
8add8B/
-
	full_text 

%136 = add nsw i64 %126, %83
&i648B

	full_text


i64 %126
%i648B

	full_text
	
i64 %83
6add8B-
+
	full_text

%137 = add nsw i64 %117, 2
&i648B

	full_text


i64 %117
8icmp8B.
,
	full_text

%138 = icmp eq i64 %137, 64
&i648B

	full_text


i64 %137
=br8B5
3
	full_text&
$
"br i1 %138, label %139, label %115
$i18B

	full_text
	
i1 %138
Bcall8	B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
<br8	B4
2
	full_text%
#
!br i1 %11, label %140, label %199
#i18	B

	full_text


i1 %11
1shl8
B(
&
	full_text

%141 = shl i64 %9, 32
$i648
B

	full_text


i64 %9
;ashr8
B1
/
	full_text"
 
%142 = ashr exact i64 %141, 32
&i648
B

	full_text


i64 %141
(br8
B 

	full_text

br label %143
Fphi8B=
;
	full_text.
,
*%144 = phi i64 [ 0, %140 ], [ %198, %195 ]
&i648B

	full_text


i64 %198
Fphi8B=
;
	full_text.
,
*%145 = phi i64 [ 1, %140 ], [ %196, %195 ]
&i648B

	full_text


i64 %196
2add8B)
'
	full_text

%146 = add i64 %144, 1
&i648B

	full_text


i64 %144
6shl8B-
+
	full_text

%147 = shl nsw i64 %145, 6
&i648B

	full_text


i64 %145
9add8B0
.
	full_text!

%148 = add nsw i64 %147, %142
&i648B

	full_text


i64 %147
&i648B

	full_text


i64 %142
^getelementptr8BK
I
	full_text<
:
8%149 = getelementptr inbounds float, float* %2, i64 %148
&i648B

	full_text


i64 %148
Nload8BD
B
	full_text5
3
1%150 = load float, float* %149, align 4, !tbaa !8
,float*8B

	full_text

float* %149
2and8B)
'
	full_text

%151 = and i64 %146, 1
&i648B

	full_text


i64 %146
7icmp8B-
+
	full_text

%152 = icmp eq i64 %144, 0
&i648B

	full_text


i64 %144
=br8B5
3
	full_text&
$
"br i1 %152, label %181, label %153
$i18B

	full_text
	
i1 %152
5sub8B,
*
	full_text

%154 = sub i64 %146, %151
&i648B

	full_text


i64 %146
&i648B

	full_text


i64 %151
(br8B 

	full_text

br label %155
Kphi8BB
@
	full_text3
1
/%156 = phi float [ %150, %153 ], [ %177, %155 ]
*float8B

	full_text


float %150
*float8B

	full_text


float %177
Fphi8B=
;
	full_text.
,
*%157 = phi i64 [ 0, %153 ], [ %178, %155 ]
&i648B

	full_text


i64 %178
Iphi8B@
>
	full_text1
/
-%158 = phi i64 [ %154, %153 ], [ %179, %155 ]
&i648B

	full_text


i64 %154
&i648B

	full_text


i64 %179
=add8B4
2
	full_text%
#
!%159 = add nuw nsw i64 %157, %147
&i648B

	full_text


i64 %157
&i648B

	full_text


i64 %147
^getelementptr8BK
I
	full_text<
:
8%160 = getelementptr inbounds float, float* %1, i64 %159
&i648B

	full_text


i64 %159
Nload8BD
B
	full_text5
3
1%161 = load float, float* %160, align 4, !tbaa !8
,float*8B

	full_text

float* %160
2shl8B)
'
	full_text

%162 = shl i64 %157, 6
&i648B

	full_text


i64 %157
9add8B0
.
	full_text!

%163 = add nsw i64 %162, %142
&i648B

	full_text


i64 %162
&i648B

	full_text


i64 %142
^getelementptr8BK
I
	full_text<
:
8%164 = getelementptr inbounds float, float* %2, i64 %163
&i648B

	full_text


i64 %163
Nload8BD
B
	full_text5
3
1%165 = load float, float* %164, align 4, !tbaa !8
,float*8B

	full_text

float* %164
Bfsub8B8
6
	full_text)
'
%%166 = fsub float -0.000000e+00, %161
*float8B

	full_text


float %161
icall8B_
]
	full_textP
N
L%167 = tail call float @llvm.fmuladd.f32(float %166, float %165, float %156)
*float8B

	full_text


float %166
*float8B

	full_text


float %165
*float8B

	full_text


float %156
Nstore8BC
A
	full_text4
2
0store float %167, float* %149, align 4, !tbaa !8
*float8B

	full_text


float %167
,float*8B

	full_text

float* %149
0or8B(
&
	full_text

%168 = or i64 %157, 1
&i648B

	full_text


i64 %157
=add8B4
2
	full_text%
#
!%169 = add nuw nsw i64 %168, %147
&i648B

	full_text


i64 %168
&i648B

	full_text


i64 %147
^getelementptr8BK
I
	full_text<
:
8%170 = getelementptr inbounds float, float* %1, i64 %169
&i648B

	full_text


i64 %169
Nload8BD
B
	full_text5
3
1%171 = load float, float* %170, align 4, !tbaa !8
,float*8B

	full_text

float* %170
2shl8B)
'
	full_text

%172 = shl i64 %168, 6
&i648B

	full_text


i64 %168
9add8B0
.
	full_text!

%173 = add nsw i64 %172, %142
&i648B

	full_text


i64 %172
&i648B

	full_text


i64 %142
^getelementptr8BK
I
	full_text<
:
8%174 = getelementptr inbounds float, float* %2, i64 %173
&i648B

	full_text


i64 %173
Nload8BD
B
	full_text5
3
1%175 = load float, float* %174, align 4, !tbaa !8
,float*8B

	full_text

float* %174
Bfsub8B8
6
	full_text)
'
%%176 = fsub float -0.000000e+00, %171
*float8B

	full_text


float %171
icall8B_
]
	full_textP
N
L%177 = tail call float @llvm.fmuladd.f32(float %176, float %175, float %167)
*float8B

	full_text


float %176
*float8B

	full_text


float %175
*float8B

	full_text


float %167
Nstore8BC
A
	full_text4
2
0store float %177, float* %149, align 4, !tbaa !8
*float8B

	full_text


float %177
,float*8B

	full_text

float* %149
6add8B-
+
	full_text

%178 = add nsw i64 %157, 2
&i648B

	full_text


i64 %157
3add8B*
(
	full_text

%179 = add i64 %158, -2
&i648B

	full_text


i64 %158
7icmp8B-
+
	full_text

%180 = icmp eq i64 %179, 0
&i648B

	full_text


i64 %179
=br8B5
3
	full_text&
$
"br i1 %180, label %181, label %155
$i18B

	full_text
	
i1 %180
Kphi8BB
@
	full_text3
1
/%182 = phi float [ %150, %143 ], [ %177, %155 ]
*float8B

	full_text


float %150
*float8B

	full_text


float %177
Fphi8B=
;
	full_text.
,
*%183 = phi i64 [ 0, %143 ], [ %178, %155 ]
&i648B

	full_text


i64 %178
7icmp8B-
+
	full_text

%184 = icmp eq i64 %151, 0
&i648B

	full_text


i64 %151
=br8B5
3
	full_text&
$
"br i1 %184, label %195, label %185
$i18B

	full_text
	
i1 %184
=add8B4
2
	full_text%
#
!%186 = add nuw nsw i64 %183, %147
&i648B

	full_text


i64 %183
&i648B

	full_text


i64 %147
^getelementptr8BK
I
	full_text<
:
8%187 = getelementptr inbounds float, float* %1, i64 %186
&i648B

	full_text


i64 %186
Nload8BD
B
	full_text5
3
1%188 = load float, float* %187, align 4, !tbaa !8
,float*8B

	full_text

float* %187
2shl8B)
'
	full_text

%189 = shl i64 %183, 6
&i648B

	full_text


i64 %183
9add8B0
.
	full_text!

%190 = add nsw i64 %189, %142
&i648B

	full_text


i64 %189
&i648B

	full_text


i64 %142
^getelementptr8BK
I
	full_text<
:
8%191 = getelementptr inbounds float, float* %2, i64 %190
&i648B

	full_text


i64 %190
Nload8BD
B
	full_text5
3
1%192 = load float, float* %191, align 4, !tbaa !8
,float*8B

	full_text

float* %191
Bfsub8B8
6
	full_text)
'
%%193 = fsub float -0.000000e+00, %188
*float8B

	full_text


float %188
icall8B_
]
	full_textP
N
L%194 = tail call float @llvm.fmuladd.f32(float %193, float %192, float %182)
*float8B

	full_text


float %193
*float8B

	full_text


float %192
*float8B

	full_text


float %182
Nstore8BC
A
	full_text4
2
0store float %194, float* %149, align 4, !tbaa !8
*float8B

	full_text


float %194
,float*8B

	full_text

float* %149
(br8B 

	full_text

br label %195
:add8B1
/
	full_text"
 
%196 = add nuw nsw i64 %145, 1
&i648B

	full_text


i64 %145
8icmp8B.
,
	full_text

%197 = icmp eq i64 %196, 64
&i648B

	full_text


i64 %196
2add8B)
'
	full_text

%198 = add i64 %144, 1
&i648B

	full_text


i64 %144
=br8B5
3
	full_text&
$
"br i1 %197, label %262, label %143
$i18B

	full_text
	
i1 %197
1shl8B(
&
	full_text

%200 = shl i32 %10, 6
%i328B

	full_text
	
i32 %10
6add8B-
+
	full_text

%201 = add i32 %200, -4096
&i328B

	full_text


i32 %200
8sext8B.
,
	full_text

%202 = sext i32 %201 to i64
&i328B

	full_text


i32 %201
(br8B 

	full_text

br label %203
Fphi8B=
;
	full_text.
,
*%204 = phi i64 [ 0, %199 ], [ %260, %254 ]
&i648B

	full_text


i64 %260
7icmp8B-
+
	full_text

%205 = icmp eq i64 %204, 0
&i648B

	full_text


i64 %204
9add8B0
.
	full_text!

%206 = add nsw i64 %204, %202
&i648B

	full_text


i64 %204
&i648B

	full_text


i64 %202
^getelementptr8BK
I
	full_text<
:
8%207 = getelementptr inbounds float, float* %3, i64 %206
&i648B

	full_text


i64 %206
=br8B5
3
	full_text&
$
"br i1 %205, label %254, label %208
$i18B

	full_text
	
i1 %205
Nload8BD
B
	full_text5
3
1%209 = load float, float* %207, align 4, !tbaa !8
,float*8B

	full_text

float* %207
2and8B)
'
	full_text

%210 = and i64 %204, 1
&i648B

	full_text


i64 %204
7icmp8B-
+
	full_text

%211 = icmp eq i64 %204, 1
&i648B

	full_text


i64 %204
=br8B5
3
	full_text&
$
"br i1 %211, label %240, label %212
$i18B

	full_text
	
i1 %211
5sub8B,
*
	full_text

%213 = sub i64 %204, %210
&i648B

	full_text


i64 %204
&i648B

	full_text


i64 %210
(br8B 

	full_text

br label %214
Kphi8BB
@
	full_text3
1
/%215 = phi float [ %209, %212 ], [ %236, %214 ]
*float8B

	full_text


float %209
*float8B

	full_text


float %236
Fphi8B=
;
	full_text.
,
*%216 = phi i64 [ 0, %212 ], [ %237, %214 ]
&i648B

	full_text


i64 %237
Iphi8B@
>
	full_text1
/
-%217 = phi i64 [ %213, %212 ], [ %238, %214 ]
&i648B

	full_text


i64 %213
&i648B

	full_text


i64 %238
=add8B4
2
	full_text%
#
!%218 = add nuw nsw i64 %216, %202
&i648B

	full_text


i64 %216
&i648B

	full_text


i64 %202
^getelementptr8BK
I
	full_text<
:
8%219 = getelementptr inbounds float, float* %3, i64 %218
&i648B

	full_text


i64 %218
Nload8BD
B
	full_text5
3
1%220 = load float, float* %219, align 4, !tbaa !8
,float*8B

	full_text

float* %219
2shl8B)
'
	full_text

%221 = shl i64 %216, 6
&i648B

	full_text


i64 %216
=add8B4
2
	full_text%
#
!%222 = add nuw nsw i64 %221, %204
&i648B

	full_text


i64 %221
&i648B

	full_text


i64 %204
^getelementptr8BK
I
	full_text<
:
8%223 = getelementptr inbounds float, float* %1, i64 %222
&i648B

	full_text


i64 %222
Nload8BD
B
	full_text5
3
1%224 = load float, float* %223, align 4, !tbaa !8
,float*8B

	full_text

float* %223
Bfsub8B8
6
	full_text)
'
%%225 = fsub float -0.000000e+00, %220
*float8B

	full_text


float %220
icall8B_
]
	full_textP
N
L%226 = tail call float @llvm.fmuladd.f32(float %225, float %224, float %215)
*float8B

	full_text


float %225
*float8B

	full_text


float %224
*float8B

	full_text


float %215
Nstore8BC
A
	full_text4
2
0store float %226, float* %207, align 4, !tbaa !8
*float8B

	full_text


float %226
,float*8B

	full_text

float* %207
0or8B(
&
	full_text

%227 = or i64 %216, 1
&i648B

	full_text


i64 %216
=add8B4
2
	full_text%
#
!%228 = add nuw nsw i64 %227, %202
&i648B

	full_text


i64 %227
&i648B

	full_text


i64 %202
^getelementptr8BK
I
	full_text<
:
8%229 = getelementptr inbounds float, float* %3, i64 %228
&i648B

	full_text


i64 %228
Nload8BD
B
	full_text5
3
1%230 = load float, float* %229, align 4, !tbaa !8
,float*8B

	full_text

float* %229
2shl8B)
'
	full_text

%231 = shl i64 %227, 6
&i648B

	full_text


i64 %227
=add8B4
2
	full_text%
#
!%232 = add nuw nsw i64 %231, %204
&i648B

	full_text


i64 %231
&i648B

	full_text


i64 %204
^getelementptr8BK
I
	full_text<
:
8%233 = getelementptr inbounds float, float* %1, i64 %232
&i648B

	full_text


i64 %232
Nload8BD
B
	full_text5
3
1%234 = load float, float* %233, align 4, !tbaa !8
,float*8B

	full_text

float* %233
Bfsub8B8
6
	full_text)
'
%%235 = fsub float -0.000000e+00, %230
*float8B

	full_text


float %230
icall8B_
]
	full_textP
N
L%236 = tail call float @llvm.fmuladd.f32(float %235, float %234, float %226)
*float8B

	full_text


float %235
*float8B

	full_text


float %234
*float8B

	full_text


float %226
Nstore8BC
A
	full_text4
2
0store float %236, float* %207, align 4, !tbaa !8
*float8B

	full_text


float %236
,float*8B

	full_text

float* %207
6add8B-
+
	full_text

%237 = add nsw i64 %216, 2
&i648B

	full_text


i64 %216
3add8B*
(
	full_text

%238 = add i64 %217, -2
&i648B

	full_text


i64 %217
7icmp8B-
+
	full_text

%239 = icmp eq i64 %238, 0
&i648B

	full_text


i64 %238
=br8B5
3
	full_text&
$
"br i1 %239, label %240, label %214
$i18B

	full_text
	
i1 %239
Kphi8BB
@
	full_text3
1
/%241 = phi float [ %209, %208 ], [ %236, %214 ]
*float8B

	full_text


float %209
*float8B

	full_text


float %236
Fphi8B=
;
	full_text.
,
*%242 = phi i64 [ 0, %208 ], [ %237, %214 ]
&i648B

	full_text


i64 %237
7icmp8B-
+
	full_text

%243 = icmp eq i64 %210, 0
&i648B

	full_text


i64 %210
=br8B5
3
	full_text&
$
"br i1 %243, label %254, label %244
$i18B

	full_text
	
i1 %243
=add8B4
2
	full_text%
#
!%245 = add nuw nsw i64 %242, %202
&i648B

	full_text


i64 %242
&i648B

	full_text


i64 %202
^getelementptr8BK
I
	full_text<
:
8%246 = getelementptr inbounds float, float* %3, i64 %245
&i648B

	full_text


i64 %245
Nload8BD
B
	full_text5
3
1%247 = load float, float* %246, align 4, !tbaa !8
,float*8B

	full_text

float* %246
2shl8B)
'
	full_text

%248 = shl i64 %242, 6
&i648B

	full_text


i64 %242
=add8B4
2
	full_text%
#
!%249 = add nuw nsw i64 %248, %204
&i648B

	full_text


i64 %248
&i648B

	full_text


i64 %204
^getelementptr8BK
I
	full_text<
:
8%250 = getelementptr inbounds float, float* %1, i64 %249
&i648B

	full_text


i64 %249
Nload8BD
B
	full_text5
3
1%251 = load float, float* %250, align 4, !tbaa !8
,float*8B

	full_text

float* %250
Bfsub8B8
6
	full_text)
'
%%252 = fsub float -0.000000e+00, %247
*float8B

	full_text


float %247
icall8B_
]
	full_textP
N
L%253 = tail call float @llvm.fmuladd.f32(float %252, float %251, float %241)
*float8B

	full_text


float %252
*float8B

	full_text


float %251
*float8B

	full_text


float %241
Nstore8BC
A
	full_text4
2
0store float %253, float* %207, align 4, !tbaa !8
*float8B

	full_text


float %253
,float*8B

	full_text

float* %207
(br8B 

	full_text

br label %254
;mul8B2
0
	full_text#
!
%255 = mul nuw nsw i64 %204, 65
&i648B

	full_text


i64 %204
^getelementptr8BK
I
	full_text<
:
8%256 = getelementptr inbounds float, float* %1, i64 %255
&i648B

	full_text


i64 %255
Nload8BD
B
	full_text5
3
1%257 = load float, float* %256, align 4, !tbaa !8
,float*8B

	full_text

float* %256
Nload8BD
B
	full_text5
3
1%258 = load float, float* %207, align 4, !tbaa !8
,float*8B

	full_text

float* %207
Ffdiv8B<
:
	full_text-
+
)%259 = fdiv float %258, %257, !fpmath !12
*float8B

	full_text


float %258
*float8B

	full_text


float %257
Nstore8BC
A
	full_text4
2
0store float %259, float* %207, align 4, !tbaa !8
*float8B

	full_text


float %259
,float*8B

	full_text

float* %207
:add8B1
/
	full_text"
 
%260 = add nuw nsw i64 %204, 1
&i648B

	full_text


i64 %204
8icmp8B.
,
	full_text

%261 = icmp eq i64 %260, 64
&i648B

	full_text


i64 %260
=br8B5
3
	full_text&
$
"br i1 %261, label %262, label %203
$i18B

	full_text
	
i1 %261
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
<br8B4
2
	full_text%
#
!br i1 %11, label %263, label %314
#i18B

	full_text


i1 %11
4add8B+
)
	full_text

%264 = add nsw i32 %5, 1
7mul8B.
,
	full_text

%265 = mul nsw i32 %264, %4
&i328B

	full_text


i32 %264
7add8B.
,
	full_text

%266 = add nsw i32 %265, %5
&i328B

	full_text


i32 %265
0shl8B'
%
	full_text

%267 = shl i32 %8, 6
$i328B

	full_text


i32 %8
3add8B*
(
	full_text

%268 = add i32 %267, 64
&i328B

	full_text


i32 %267
4add8B+
)
	full_text

%269 = add i32 %268, %10
&i328B

	full_text


i32 %268
%i328B

	full_text
	
i32 %10
1shl8B(
&
	full_text

%270 = shl i64 %9, 32
$i648B

	full_text


i64 %9
;ashr8B1
/
	full_text"
 
%271 = ashr exact i64 %270, 32
&i648B

	full_text


i64 %270
8sext8B.
,
	full_text

%272 = sext i32 %266 to i64
&i328B

	full_text


i32 %266
6sext8B,
*
	full_text

%273 = sext i32 %4 to i64
(br8B 

	full_text

br label %274
Iphi8B@
>
	full_text1
/
-%275 = phi i64 [ %272, %263 ], [ %311, %274 ]
&i648B

	full_text


i64 %272
&i648B

	full_text


i64 %311
Fphi8B=
;
	full_text.
,
*%276 = phi i64 [ 1, %263 ], [ %312, %274 ]
&i648B

	full_text


i64 %312
2shl8B)
'
	full_text

%277 = shl i64 %276, 6
&i648B

	full_text


i64 %276
9add8B0
.
	full_text!

%278 = add nsw i64 %277, %271
&i648B

	full_text


i64 %277
&i648B

	full_text


i64 %271
^getelementptr8BK
I
	full_text<
:
8%279 = getelementptr inbounds float, float* %2, i64 %278
&i648B

	full_text


i64 %278
Bbitcast8B5
3
	full_text&
$
"%280 = bitcast float* %279 to i32*
,float*8B

	full_text

float* %279
Jload8B@
>
	full_text1
/
-%281 = load i32, i32* %280, align 4, !tbaa !8
(i32*8B

	full_text

	i32* %280
:trunc8B/
-
	full_text 

%282 = trunc i64 %275 to i32
&i648B

	full_text


i64 %275
5add8B,
*
	full_text

%283 = add i32 %269, %282
&i328B

	full_text


i32 %269
&i328B

	full_text


i32 %282
8sext8B.
,
	full_text

%284 = sext i32 %283 to i64
&i328B

	full_text


i32 %283
^getelementptr8BK
I
	full_text<
:
8%285 = getelementptr inbounds float, float* %0, i64 %284
&i648B

	full_text


i64 %284
Bbitcast8B5
3
	full_text&
$
"%286 = bitcast float* %285 to i32*
,float*8B

	full_text

float* %285
Jstore8B?
=
	full_text0
.
,store i32 %281, i32* %286, align 4, !tbaa !8
&i328B

	full_text


i32 %281
(i32*8B

	full_text

	i32* %286
9add8B0
.
	full_text!

%287 = add nsw i64 %275, %273
&i648B

	full_text


i64 %275
&i648B

	full_text


i64 %273
2shl8B)
'
	full_text

%288 = shl i64 %276, 6
&i648B

	full_text


i64 %276
3add8B*
(
	full_text

%289 = add i64 %288, 64
&i648B

	full_text


i64 %288
9add8B0
.
	full_text!

%290 = add nsw i64 %289, %271
&i648B

	full_text


i64 %289
&i648B

	full_text


i64 %271
^getelementptr8BK
I
	full_text<
:
8%291 = getelementptr inbounds float, float* %2, i64 %290
&i648B

	full_text


i64 %290
Bbitcast8B5
3
	full_text&
$
"%292 = bitcast float* %291 to i32*
,float*8B

	full_text

float* %291
Jload8B@
>
	full_text1
/
-%293 = load i32, i32* %292, align 4, !tbaa !8
(i32*8B

	full_text

	i32* %292
:trunc8B/
-
	full_text 

%294 = trunc i64 %287 to i32
&i648B

	full_text


i64 %287
5add8B,
*
	full_text

%295 = add i32 %269, %294
&i328B

	full_text


i32 %269
&i328B

	full_text


i32 %294
8sext8B.
,
	full_text

%296 = sext i32 %295 to i64
&i328B

	full_text


i32 %295
^getelementptr8BK
I
	full_text<
:
8%297 = getelementptr inbounds float, float* %0, i64 %296
&i648B

	full_text


i64 %296
Bbitcast8B5
3
	full_text&
$
"%298 = bitcast float* %297 to i32*
,float*8B

	full_text

float* %297
Jstore8B?
=
	full_text0
.
,store i32 %293, i32* %298, align 4, !tbaa !8
&i328B

	full_text


i32 %293
(i32*8B

	full_text

	i32* %298
9add8B0
.
	full_text!

%299 = add nsw i64 %287, %273
&i648B

	full_text


i64 %287
&i648B

	full_text


i64 %273
2shl8B)
'
	full_text

%300 = shl i64 %276, 6
&i648B

	full_text


i64 %276
4add8B+
)
	full_text

%301 = add i64 %300, 128
&i648B

	full_text


i64 %300
9add8B0
.
	full_text!

%302 = add nsw i64 %301, %271
&i648B

	full_text


i64 %301
&i648B

	full_text


i64 %271
^getelementptr8BK
I
	full_text<
:
8%303 = getelementptr inbounds float, float* %2, i64 %302
&i648B

	full_text


i64 %302
Bbitcast8B5
3
	full_text&
$
"%304 = bitcast float* %303 to i32*
,float*8B

	full_text

float* %303
Jload8B@
>
	full_text1
/
-%305 = load i32, i32* %304, align 4, !tbaa !8
(i32*8B

	full_text

	i32* %304
:trunc8B/
-
	full_text 

%306 = trunc i64 %299 to i32
&i648B

	full_text


i64 %299
5add8B,
*
	full_text

%307 = add i32 %269, %306
&i328B

	full_text


i32 %269
&i328B

	full_text


i32 %306
8sext8B.
,
	full_text

%308 = sext i32 %307 to i64
&i328B

	full_text


i32 %307
^getelementptr8BK
I
	full_text<
:
8%309 = getelementptr inbounds float, float* %0, i64 %308
&i648B

	full_text


i64 %308
Bbitcast8B5
3
	full_text&
$
"%310 = bitcast float* %309 to i32*
,float*8B

	full_text

float* %309
Jstore8B?
=
	full_text0
.
,store i32 %305, i32* %310, align 4, !tbaa !8
&i328B

	full_text


i32 %305
(i32*8B

	full_text

	i32* %310
9add8B0
.
	full_text!

%311 = add nsw i64 %299, %273
&i648B

	full_text


i64 %299
&i648B

	full_text


i64 %273
6add8B-
+
	full_text

%312 = add nsw i64 %276, 3
&i648B

	full_text


i64 %276
8icmp8B.
,
	full_text

%313 = icmp eq i64 %312, 64
&i648B

	full_text


i64 %312
=br8B5
3
	full_text&
$
"br i1 %313, label %349, label %274
$i18B

	full_text
	
i1 %313
0shl8B'
%
	full_text

%315 = shl i32 %8, 6
$i328B

	full_text


i32 %8
1add8B(
&
	full_text

%316 = add i32 %5, 64
5add8B,
*
	full_text

%317 = add i32 %316, %315
&i328B

	full_text


i32 %316
&i328B

	full_text


i32 %315
7mul8B.
,
	full_text

%318 = mul nsw i32 %317, %4
&i328B

	full_text


i32 %317
7add8B.
,
	full_text

%319 = add nsw i32 %318, %5
&i328B

	full_text


i32 %318
1shl8B(
&
	full_text

%320 = shl i64 %9, 32
$i648B

	full_text


i64 %9
>add8B5
3
	full_text&
$
"%321 = add i64 %320, -274877906944
&i648B

	full_text


i64 %320
;ashr8B1
/
	full_text"
 
%322 = ashr exact i64 %321, 32
&i648B

	full_text


i64 %321
8sext8B.
,
	full_text

%323 = sext i32 %319 to i64
&i328B

	full_text


i32 %319
6sext8B,
*
	full_text

%324 = sext i32 %4 to i64
(br8B 

	full_text

br label %325
Iphi8B@
>
	full_text1
/
-%326 = phi i64 [ %323, %314 ], [ %346, %325 ]
&i648B

	full_text


i64 %323
&i648B

	full_text


i64 %346
Fphi8B=
;
	full_text.
,
*%327 = phi i64 [ 0, %314 ], [ %347, %325 ]
&i648B

	full_text


i64 %347
2shl8B)
'
	full_text

%328 = shl i64 %327, 6
&i648B

	full_text


i64 %327
9add8B0
.
	full_text!

%329 = add nsw i64 %328, %322
&i648B

	full_text


i64 %328
&i648B

	full_text


i64 %322
^getelementptr8BK
I
	full_text<
:
8%330 = getelementptr inbounds float, float* %3, i64 %329
&i648B

	full_text


i64 %329
Bbitcast8B5
3
	full_text&
$
"%331 = bitcast float* %330 to i32*
,float*8B

	full_text

float* %330
Jload8B@
>
	full_text1
/
-%332 = load i32, i32* %331, align 4, !tbaa !8
(i32*8B

	full_text

	i32* %331
9add8B0
.
	full_text!

%333 = add nsw i64 %326, %322
&i648B

	full_text


i64 %326
&i648B

	full_text


i64 %322
^getelementptr8BK
I
	full_text<
:
8%334 = getelementptr inbounds float, float* %0, i64 %333
&i648B

	full_text


i64 %333
Bbitcast8B5
3
	full_text&
$
"%335 = bitcast float* %334 to i32*
,float*8B

	full_text

float* %334
Jstore8B?
=
	full_text0
.
,store i32 %332, i32* %335, align 4, !tbaa !8
&i328B

	full_text


i32 %332
(i32*8B

	full_text

	i32* %335
9add8B0
.
	full_text!

%336 = add nsw i64 %326, %324
&i648B

	full_text


i64 %326
&i648B

	full_text


i64 %324
2shl8B)
'
	full_text

%337 = shl i64 %327, 6
&i648B

	full_text


i64 %327
1or8B)
'
	full_text

%338 = or i64 %337, 64
&i648B

	full_text


i64 %337
9add8B0
.
	full_text!

%339 = add nsw i64 %338, %322
&i648B

	full_text


i64 %338
&i648B

	full_text


i64 %322
^getelementptr8BK
I
	full_text<
:
8%340 = getelementptr inbounds float, float* %3, i64 %339
&i648B

	full_text


i64 %339
Bbitcast8B5
3
	full_text&
$
"%341 = bitcast float* %340 to i32*
,float*8B

	full_text

float* %340
Jload8B@
>
	full_text1
/
-%342 = load i32, i32* %341, align 4, !tbaa !8
(i32*8B

	full_text

	i32* %341
9add8B0
.
	full_text!

%343 = add nsw i64 %336, %322
&i648B

	full_text


i64 %336
&i648B

	full_text


i64 %322
^getelementptr8BK
I
	full_text<
:
8%344 = getelementptr inbounds float, float* %0, i64 %343
&i648B

	full_text


i64 %343
Bbitcast8B5
3
	full_text&
$
"%345 = bitcast float* %344 to i32*
,float*8B

	full_text

float* %344
Jstore8B?
=
	full_text0
.
,store i32 %342, i32* %345, align 4, !tbaa !8
&i328B

	full_text


i32 %342
(i32*8B

	full_text

	i32* %345
9add8B0
.
	full_text!

%346 = add nsw i64 %336, %324
&i648B

	full_text


i64 %336
&i648B

	full_text


i64 %324
6add8B-
+
	full_text

%347 = add nsw i64 %327, 2
&i648B

	full_text


i64 %327
8icmp8B.
,
	full_text

%348 = icmp eq i64 %347, 64
&i648B

	full_text


i64 %347
=br8B5
3
	full_text&
$
"br i1 %348, label %349, label %325
$i18B

	full_text
	
i1 %348
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %4
*float*8B

	full_text

	float* %0
$i328B

	full_text


i32 %5
*float*8B

	full_text

	float* %3
*float*8B

	full_text

	float* %2
*float*8B
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
#i328B

	full_text	

i32 6
#i648B

	full_text	

i64 2
#i328B

	full_text	

i32 0
$i328B

	full_text


i32 32
#i328B

	full_text	

i32 1
3float8B&
$
	full_text

float -0.000000e+00
$i648B

	full_text


i64 32
#i648B

	full_text	

i64 3
/i648B$
"
	full_text

i64 -274877906944
$i648B

	full_text


i64 65
%i648B

	full_text
	
i64 128
#i648B

	full_text	

i64 0
#i648B

	full_text	

i64 6
$i648B

	full_text


i64 64
#i648B

	full_text	

i64 1
$i328B

	full_text


i32 64
$i648B

	full_text


i64 -2
'i328B

	full_text

	i32 -4096        	
 	                     !    "# "" $% $$ &' &( && )* )) +, ++ -. -/ -- 01 02 00 34 35 33 67 66 89 88 :; :: <= << >? >> @A @B @@ CD CC EF EE GH GI GG JK JL JJ MN MM OP OO QR QT SS UV UU WX WY WW Z\ [] [[ ^_ ^^ `a `` bc bd bb ef ee gh gg ij ii kl kk mn mm op oq oo rs rr tu tt vw vx vv yz y{ yy |} || ~ ~	€ ~~ ‚  ƒ
„ ƒƒ …† …… ‡ˆ ‡‡ ‰Š ‰‰ ‹Œ ‹‹ Ž 
  
‘  ’“ ’’ ”• ”
– ”” —˜ —
™ —— š› šš œ œœ žŸ ž  ¡¢ ¡¡ £¤ ££ ¥¦ ¥¥ §¨ §§ ©ª ©© «¬ «« ­­ ®° ¯
± ¯¯ ²
³ ²² ´µ ´
¶ ´´ ·
¸ ·· ¹º ¹¹ »¼ »» ½¾ ½½ ¿À ¿
Á ¿¿ Â
Ã ÂÂ ÄÅ ÄÄ ÆÇ Æ
È ÆÆ ÉÊ É
Ë ÉÉ ÌÍ Ì
Î ÌÌ Ï
Ð ÏÏ ÑÒ ÑÑ ÓÔ ÓÓ ÕÖ ÕÕ ×Ø ×× ÙÚ Ù
Û ÙÙ Ü
Ý ÜÜ Þß ÞÞ àá à
â àà ãä ã
å ãã æç ææ èé èè êë êí ìì îî ïð ï
ñ ïï òó òò ôõ ôô ö÷ öö øú ù
û ùù ü
ý üü þÿ þ
€ þþ 
‚  ƒ„ ƒƒ …† …… ‡ˆ ‡‡ ‰Š ‰
‹ ‰‰ Œ
 ŒŒ Ž ŽŽ ‘ 
’  “” “
• ““ –— –
˜ –– ™
š ™™ ›œ ›› ž  Ÿ  ŸŸ ¡¢ ¡¡ £¤ £
¥ ££ ¦
§ ¦¦ ¨© ¨¨ ª« ª
¬ ªª ­® ­
¯ ­­ °± °° ²³ ²² ´µ ´¶ ·¸ ·º ¹¹ »¼ »» ½
¿ ¾¾ À
Á ÀÀ ÂÃ ÂÂ ÄÅ ÄÄ ÆÇ Æ
È ÆÆ É
Ê ÉÉ ËÌ ËË ÍÎ ÍÍ ÏÐ ÏÏ ÑÒ ÑÔ Ó
Õ ÓÓ ÖØ ×
Ù ×× Ú
Û ÚÚ ÜÝ Ü
Þ ÜÜ ßà ß
á ßß â
ã ââ äå ää æç ææ èé è
ê èè ë
ì ëë íî íí ï
ð ïï ñò ñ
ó ñ
ô ññ õö õ
÷ õõ øù øø úû ú
ü úú ý
þ ýý ÿ€ ÿÿ ‚  ƒ„ ƒ
… ƒƒ †
‡ †† ˆ‰ ˆˆ Š
‹ ŠŠ Œ Œ
Ž Œ
 ŒŒ ‘ 
’  “” ““ •– •• —˜ —— ™š ™œ ›
 ›› ž
Ÿ žž  ¡    ¢£ ¢¥ ¤
¦ ¤¤ §
¨ §§ ©ª ©© «¬ «« ­® ­
¯ ­­ °
± °° ²³ ²² ´
µ ´´ ¶· ¶
¸ ¶
¹ ¶¶ º» º
¼ ºº ½¿ ¾¾ ÀÁ ÀÀ ÂÃ ÂÂ ÄÅ ÄÇ ÆÆ ÈÉ ÈÈ ÊË ÊÊ Ì
Î ÍÍ ÏÐ ÏÏ ÑÒ Ñ
Ó ÑÑ Ô
Õ ÔÔ Ö× ÖÙ ØØ ÚÛ ÚÚ ÜÝ ÜÜ Þß Þá à
â àà ãå ä
æ ää ç
è çç éê é
ë éé ìí ì
î ìì ï
ð ïï ñò ññ óô óó õö õ
÷ õõ ø
ù øø úû úú ü
ý üü þÿ þ
€ þ
 þþ ‚ƒ ‚
„ ‚‚ …† …… ‡ˆ ‡
‰ ‡‡ Š
‹ ŠŠ Œ ŒŒ Ž ŽŽ ‘ 
’  “
” ““ •– •• —
˜ —— ™š ™
› ™
œ ™™ ž 
Ÿ   ¡    ¢£ ¢¢ ¤¥ ¤¤ ¦§ ¦© ¨
ª ¨¨ «
¬ «« ­® ­­ ¯° ¯² ±
³ ±± ´
µ ´´ ¶· ¶¶ ¸¹ ¸¸ º» º
¼ ºº ½
¾ ½½ ¿À ¿¿ Á
Â ÁÁ ÃÄ Ã
Å Ã
Æ ÃÃ ÇÈ Ç
É ÇÇ ÊÌ ËË Í
Î ÍÍ ÏÐ ÏÏ ÑÒ ÑÑ ÓÔ Ó
Õ ÓÓ Ö× Ö
Ø ÖÖ ÙÚ ÙÙ ÛÜ ÛÛ ÝÞ Ýß àá àâ ãä ãã åæ åå çè çç éê éé ëì ë
í ëë îï îî ðñ ðð òó òò ôô õ÷ ö
ø öö ù
ú ùù ûü ûû ýþ ý
ÿ ýý €
 €€ ‚ƒ ‚‚ „… „„ †‡ †† ˆ‰ ˆ
Š ˆˆ ‹Œ ‹‹ 
Ž    ‘’ ‘
“ ‘‘ ”• ”
– ”” —˜ —— ™š ™™ ›œ ›
 ›› ž
Ÿ žž  ¡    ¢£ ¢¢ ¤¥ ¤¤ ¦§ ¦
¨ ¦¦ ©ª ©© «
¬ «« ­® ­­ ¯° ¯
± ¯¯ ²³ ²
´ ²² µ¶ µµ ·¸ ·· ¹º ¹
» ¹¹ ¼
½ ¼¼ ¾¿ ¾¾ ÀÁ ÀÀ ÂÃ ÂÂ ÄÅ Ä
Æ ÄÄ ÇÈ ÇÇ É
Ê ÉÉ ËÌ ËË ÍÎ Í
Ï ÍÍ ÐÑ Ð
Ò ÐÐ ÓÔ ÓÓ ÕÖ ÕÕ ×Ø ×Ú ÙÙ ÛÛ ÜÝ Ü
Þ ÜÜ ßà ßß áâ áá ãä ãã åæ åå çè çç éê éé ëë ìî í
ï íí ð
ñ ðð òó òò ôõ ô
ö ôô ÷
ø ÷÷ ùú ùù ûü ûû ýþ ý
ÿ ýý €
 €€ ‚ƒ ‚‚ „… „
† „„ ‡ˆ ‡
‰ ‡‡ Š‹ ŠŠ Œ ŒŒ Ž Ž
 ŽŽ ‘
’ ‘‘ “” ““ •– •• —˜ —
™ —— š
› šš œ œœ žŸ ž
  žž ¡¢ ¡
£ ¡¡ ¤¥ ¤¤ ¦§ ¦¦ ¨© ¨	« « 
« ¡« ­
« ò
« ã« ô
« ß« ë¬ ¬ 6¬ g¬ ƒ¬ ·¬ Ï¬ ¬ ™¬ ¬ «¬ É¬ €¬ š­ 	­ ­  
­ £­ î
­ ô­ â
­ å­ Û
­ á® Œ® ¦® Ô® ï® Š® ´® ÷® ‘¯ r¯ ¯ É¯ ë¯ †¯ °¯ €¯ ž¯ ¼° )° C° Â° Ü° â° ý° §° ø° “° ½° Í    
     J M     !  # %$ ' (& *) ," .+ / 1 20 4 53 76 98 ; =< ?> A B@ DC F: HE I0 K L NM PO R TS VU X Y \— ]š _[ aW c` db fe hg ji l^ nm p qo sr uk wt x[ z {y }W | €~ ‚ „ƒ †… ˆ^ Š‰ Œ‹ Ž  ‘ “‡ •’ –y ˜ ™^ ›š œ Ÿ  ¢¡ ¤ ¦¥ ¨§ ª£ ¬« °ã ±æ ³¯ µ© ¶´ ¸· º¹ ¼² ¾½ À© Á¿ ÃÂ Å» ÇÄ È¯ Ê­ ËÉ Í© ÎÌ ÐÏ ÒÑ Ô² ÖÕ Ø× Ú© ÛÙ ÝÜ ßÓ áÞ âÉ ä­ å² çæ éè ë íî ðì ñï óò õô ÷ö ú­ û° ýù ÿ© €þ ‚ „ƒ †ü ˆ‡ Š© ‹‰ Œ … ‘Ž ’ù ”­ •“ —© ˜– š™ œ› žü  Ÿ ¢¡ ¤© ¥£ §¦ © «¨ ¬“ ®­ ¯ü ±° ³² µ ¸ º¹ ¼Â ¿¾ Á¾ ÃÀ ÅÄ Ç» ÈÆ ÊÉ ÌÂ Î¾ ÐÏ ÒÂ ÔÍ ÕË ØŒ Ù“ ÛÓ Ý• ÞÚ àÄ áß ãâ åÚ çæ é» êè ìë îä ðï òí ó× ôñ öÉ ÷Ú ùø ûÄ üú þý €ø ‚ „» …ƒ ‡† ‰ÿ ‹Š ˆ Žñ Œ ‘É ’Ú ”Ü –• ˜— šË œŒ “ ŸÍ ¡  £ž ¥Ä ¦¤ ¨§ ªž ¬« ®» ¯­ ±° ³© µ´ ·² ¸› ¹¶ »É ¼À ¿¾ Á¾ ÃÀ Å ÇÆ ÉÈ ËÙ ÎÍ ÐÍ ÒÊ ÓÑ ÕÏ ×Ô ÙÍ ÛÍ ÝÜ ßÍ áÚ âØ å™ æ  èà ê¢ ëç íÊ îì ðï òç ôó öÍ ÷õ ùø ûñ ýü ÿú €ä þ ƒÔ „ç †… ˆÊ ‰‡ ‹Š … Ž ‘Í ’ ”“ –Œ ˜— š• ›þ œ™ žÔ Ÿç ¡é £¢ ¥¤ §Ø ©™ ª  ¬Ú ®­ °« ²Ê ³± µ´ ·« ¹¸ »Í ¼º ¾½ À¶ ÂÁ Ä¿ Å¨ ÆÃ ÈÔ ÉÍ ÌË ÎÍ ÐÔ ÒÑ ÔÏ ÕÓ ×Ô ØÍ ÚÙ ÜÛ Þ áâ äã æ èç êé ì í ïî ñå óò ÷Ð øÓ úù üû þð ÿý € ƒ‚ …ö ‡ë ‰† Šˆ Œ‹ Ž „ ’ “ö •ô –ù ˜— š™ œð › Ÿž ¡  £” ¥ë §¤ ¨¦ ª© ¬« ®¢ °­ ±” ³ô ´ù ¶µ ¸· ºð »¹ ½¼ ¿¾ Á² Ãë ÅÂ ÆÄ ÈÇ ÊÉ ÌÀ ÎË Ï² Ñô Òù ÔÓ ÖÕ Ø ÚÛ ÝÙ ÞÜ àß â äã æå èá êé î¡ ï¤ ñð óò õç öô ø÷ úù üí þç ÿý € ƒû …‚ †í ˆë ‰ð ‹Š Œ ç Ž ’‘ ”“ –‡ ˜ç ™— ›š • Ÿœ  ‡ ¢ë £ð ¥¤ §¦ ©	 	   ® ¯Q SQ ê ìê ¯Z [ø ùž ¶ž [´ ¶´ ù· ¹· Æ½ ¾Ì ÍÑ ›Ñ ÓÖ ËÖ Ø¢ ¾¢ ¤Ö ×Ý ßÝ ÍÞ ¨Þ àÄ ßÄ ¾½ ¾™ ›™ ×à âà Ù¯ Ë¯ ±ã äõ öì íÊ Ë¦ ¨¦ ä× ª× ö¨ ª¨ í ±± ²² ³³ ª ´´ ²² ß ³³ ßþ ´´ þŒ ´´ ŒÃ ´´ Ã ±± ¶ ³³ ¶¶ ´´ ¶ñ ´´ ñ™ ´´ ™	µ S
µ ì
µ Æ
µ ç
µ Ù	¶ M
¶ š
¶ æ
¶ °
¶ “
¶  
¶ ¤· · 
¸  ¹ ¶¹ ß
¹ âº ïº Šº ´º üº —º Á	» 	» 	» O
» ¥
» ©» ²
» ¹
» »
» î
» ð
» ã
» ç
¼ Ó
½ §
½ å
¾ Ë
¿ ·À À ^À üÀ ¾
À ÏÀ Ú
À —À ž
À  À Í
À ÏÀ ç
À ¤À «
À ­À ð	Á $	Á <	Á m
Á ‰
Á ½
Á Õ
Á ‡
Á Ÿ
Á Ä
Á æ
Á 
Á «
Á ó
Á Ž
Á ¸
Á û
Á —
Á µ
Á ò
Á Š	Â >
Â ‹
Â œ
Â ×
Â è
Â ¡
Â ²
Â À
Â Û
Â ™
Â Õ
Â Œ
Â ¦Ã À
Ã Â
Ã Í
Ã ø
Ã ¾
Ã Â
Ã Ú
Ã Ü
Ã …
Ã ÙÃ ù	Ä 	Ä U
Ä î
Ä é
Ä Û
Å •
Å ¢
Æ È"
lud_perimeter"
_Z12get_group_idj"
_Z12get_local_idj"
_Z7barrierj"
llvm.fmuladd.f32*™
 rodinia-3.1-lud-lud_perimeter.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282€
 
transfer_bytes_log1p
áüsA

wgsize_log1p
áüsA

transfer_bytes
€€€

devmap_label
 

wgsize
 