

[external]
KcallBC
A
	full_text4
2
0%13 = tail call i64 @_Z12get_group_idj(i32 0) #4
6truncB-
+
	full_text

%14 = trunc i64 %13 to i32
#i64B

	full_text
	
i64 %13
KcallBC
A
	full_text4
2
0%15 = tail call i64 @_Z12get_local_idj(i32 0) #4
6truncB-
+
	full_text

%16 = trunc i64 %15 to i32
#i64B

	full_text
	
i64 %15
.subB'
%
	full_text

%17 = sub i32 %8, %7
0addB)
'
	full_text

%18 = add i32 %17, %14
#i32B

	full_text
	
i32 %17
#i32B

	full_text
	
i32 %14
4addB-
+
	full_text

%19 = add i32 %8, 67108863
0subB)
'
	full_text

%20 = sub i32 %19, %14
#i32B

	full_text
	
i32 %19
#i32B

	full_text
	
i32 %14
.shlB'
%
	full_text

%21 = shl i32 %20, 6
#i32B

	full_text
	
i32 %20
2shlB+
)
	full_text

%22 = shl nsw i32 %18, 6
#i32B

	full_text
	
i32 %18
0addB)
'
	full_text

%23 = add i32 %21, %10
#i32B

	full_text
	
i32 %21
/mulB(
&
	full_text

%24 = mul i32 %23, %5
#i32B

	full_text
	
i32 %23
0addB)
'
	full_text

%25 = add i32 %22, %11
#i32B

	full_text
	
i32 %22
0addB)
'
	full_text

%26 = add i32 %25, %24
#i32B

	full_text
	
i32 %25
#i32B

	full_text
	
i32 %24
4addB-
+
	full_text

%27 = add nsw i32 %26, %16
#i32B

	full_text
	
i32 %26
#i32B

	full_text
	
i32 %16
1addB*
(
	full_text

%28 = add nsw i32 %5, 1
4addB-
+
	full_text

%29 = add nsw i32 %28, %27
#i32B

	full_text
	
i32 %28
#i32B

	full_text
	
i32 %27
2addB+
)
	full_text

%30 = add nsw i32 %27, 1
#i32B

	full_text
	
i32 %27
3icmpB+
)
	full_text

%31 = icmp eq i32 %16, 0
#i32B

	full_text
	
i32 %16
8brB2
0
	full_text#
!
br i1 %31, label %32, label %39
!i1B

	full_text


i1 %31
6sext8B,
*
	full_text

%33 = sext i32 %26 to i64
%i328B

	full_text
	
i32 %26
Xgetelementptr8BE
C
	full_text6
4
2%34 = getelementptr inbounds i32, i32* %1, i64 %33
%i648B

	full_text
	
i64 %33
Hload8B>
<
	full_text/
-
+%35 = load i32, i32* %34, align 4, !tbaa !8
'i32*8B

	full_text


i32* %34
;mul8B2
0
	full_text#
!
%36 = mul i64 %15, 279172874240
%i648B

	full_text
	
i64 %15
9ashr8B/
-
	full_text 

%37 = ashr exact i64 %36, 32
%i648B

	full_text
	
i64 %36
Xgetelementptr8BE
C
	full_text6
4
2%38 = getelementptr inbounds i32, i32* %3, i64 %37
%i648B

	full_text
	
i64 %37
Hstore8B=
;
	full_text.
,
*store i32 %35, i32* %38, align 4, !tbaa !8
%i328B

	full_text
	
i32 %35
'i32*8B

	full_text


i32* %38
'br8B

	full_text

br label %39
5sext8B+
)
	full_text

%40 = sext i32 %5 to i64
6sext8B,
*
	full_text

%41 = sext i32 %29 to i64
%i328B

	full_text
	
i32 %29
1shl8B(
&
	full_text

%42 = shl i64 %15, 32
%i648B

	full_text
	
i64 %15
9ashr8B/
-
	full_text 

%43 = ashr exact i64 %42, 32
%i648B

	full_text
	
i64 %42
'br8B

	full_text

br label %60
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
5mul8B,
*
	full_text

%45 = mul nsw i32 %16, %5
%i328B

	full_text
	
i32 %16
1add8B(
&
	full_text

%46 = add i32 %45, %5
%i328B

	full_text
	
i32 %45
2add8B)
'
	full_text

%47 = add i32 %46, %26
%i328B

	full_text
	
i32 %46
%i328B

	full_text
	
i32 %26
6sext8B,
*
	full_text

%48 = sext i32 %47 to i64
%i328B

	full_text
	
i32 %47
Xgetelementptr8BE
C
	full_text6
4
2%49 = getelementptr inbounds i32, i32* %1, i64 %48
%i648B

	full_text
	
i64 %48
Hload8B>
<
	full_text/
-
+%50 = load i32, i32* %49, align 4, !tbaa !8
'i32*8B

	full_text


i32* %49
4add8B+
)
	full_text

%51 = add nsw i32 %16, 1
%i328B

	full_text
	
i32 %16
5mul8B,
*
	full_text

%52 = mul nsw i32 %51, 65
%i328B

	full_text
	
i32 %51
6sext8B,
*
	full_text

%53 = sext i32 %52 to i64
%i328B

	full_text
	
i32 %52
Xgetelementptr8BE
C
	full_text6
4
2%54 = getelementptr inbounds i32, i32* %3, i64 %53
%i648B

	full_text
	
i64 %53
Hstore8B=
;
	full_text.
,
*store i32 %50, i32* %54, align 4, !tbaa !8
%i328B

	full_text
	
i32 %50
'i32*8B

	full_text


i32* %54
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
6sext8B,
*
	full_text

%55 = sext i32 %30 to i64
%i328B

	full_text
	
i32 %30
Xgetelementptr8BE
C
	full_text6
4
2%56 = getelementptr inbounds i32, i32* %1, i64 %55
%i648B

	full_text
	
i64 %55
Hload8B>
<
	full_text/
-
+%57 = load i32, i32* %56, align 4, !tbaa !8
'i32*8B

	full_text


i32* %56
6sext8B,
*
	full_text

%58 = sext i32 %51 to i64
%i328B

	full_text
	
i32 %51
Xgetelementptr8BE
C
	full_text6
4
2%59 = getelementptr inbounds i32, i32* %3, i64 %58
%i648B

	full_text
	
i64 %58
Hstore8B=
;
	full_text.
,
*store i32 %57, i32* %59, align 4, !tbaa !8
%i328B

	full_text
	
i32 %57
'i32*8B

	full_text


i32* %59
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
'br8B

	full_text

br label %86
Bphi8B9
7
	full_text*
(
&%61 = phi i64 [ 0, %39 ], [ %77, %60 ]
%i648B

	full_text
	
i64 %77
6mul8B-
+
	full_text

%62 = mul nsw i64 %61, %40
%i648B

	full_text
	
i64 %61
%i648B

	full_text
	
i64 %40
6add8B-
+
	full_text

%63 = add nsw i64 %62, %41
%i648B

	full_text
	
i64 %62
%i648B

	full_text
	
i64 %41
Xgetelementptr8BE
C
	full_text6
4
2%64 = getelementptr inbounds i32, i32* %0, i64 %63
%i648B

	full_text
	
i64 %63
Hload8B>
<
	full_text/
-
+%65 = load i32, i32* %64, align 4, !tbaa !8
'i32*8B

	full_text


i32* %64
0shl8B'
%
	full_text

%66 = shl i64 %61, 6
%i648B

	full_text
	
i64 %61
6add8B-
+
	full_text

%67 = add nsw i64 %66, %43
%i648B

	full_text
	
i64 %66
%i648B

	full_text
	
i64 %43
Xgetelementptr8BE
C
	full_text6
4
2%68 = getelementptr inbounds i32, i32* %4, i64 %67
%i648B

	full_text
	
i64 %67
Hstore8B=
;
	full_text.
,
*store i32 %65, i32* %68, align 4, !tbaa !8
%i328B

	full_text
	
i32 %65
'i32*8B

	full_text


i32* %68
.or8B&
$
	full_text

%69 = or i64 %61, 1
%i648B

	full_text
	
i64 %61
6mul8B-
+
	full_text

%70 = mul nsw i64 %69, %40
%i648B

	full_text
	
i64 %69
%i648B

	full_text
	
i64 %40
6add8B-
+
	full_text

%71 = add nsw i64 %70, %41
%i648B

	full_text
	
i64 %70
%i648B

	full_text
	
i64 %41
Xgetelementptr8BE
C
	full_text6
4
2%72 = getelementptr inbounds i32, i32* %0, i64 %71
%i648B

	full_text
	
i64 %71
Hload8B>
<
	full_text/
-
+%73 = load i32, i32* %72, align 4, !tbaa !8
'i32*8B

	full_text


i32* %72
0shl8B'
%
	full_text

%74 = shl i64 %69, 6
%i648B

	full_text
	
i64 %69
6add8B-
+
	full_text

%75 = add nsw i64 %74, %43
%i648B

	full_text
	
i64 %74
%i648B

	full_text
	
i64 %43
Xgetelementptr8BE
C
	full_text6
4
2%76 = getelementptr inbounds i32, i32* %4, i64 %75
%i648B

	full_text
	
i64 %75
Hstore8B=
;
	full_text.
,
*store i32 %73, i32* %76, align 4, !tbaa !8
%i328B

	full_text
	
i32 %73
'i32*8B

	full_text


i32* %76
4add8B+
)
	full_text

%77 = add nsw i64 %61, 2
%i648B

	full_text
	
i64 %61
6icmp8B,
*
	full_text

%78 = icmp eq i64 %77, 64
%i648B

	full_text
	
i64 %77
:br8B2
0
	full_text#
!
br i1 %78, label %44, label %60
#i18B

	full_text


i1 %78
5add8B,
*
	full_text

%80 = add nsw i32 %16, 64
%i328B

	full_text
	
i32 %16
5sub8B,
*
	full_text

%81 = sub nsw i32 64, %16
%i328B

	full_text
	
i32 %16
5add8B,
*
	full_text

%82 = add nsw i32 %81, -1
%i328B

	full_text
	
i32 %81
5mul8B,
*
	full_text

%83 = mul nsw i32 %82, 65
%i328B

	full_text
	
i32 %82
0shl8B'
%
	full_text

%84 = shl i32 %82, 6
%i328B

	full_text
	
i32 %82
5mul8B,
*
	full_text

%85 = mul nsw i32 %81, 65
%i328B

	full_text
	
i32 %81
(br8B 

	full_text

br label %121
Dphi8B;
9
	full_text,
*
(%87 = phi i64 [ 0, %44 ], [ %119, %118 ]
&i648B

	full_text


i64 %119
8icmp8B.
,
	full_text

%88 = icmp slt i64 %87, %43
%i648B

	full_text
	
i64 %87
%i648B

	full_text
	
i64 %43
;br8B3
1
	full_text$
"
 br i1 %88, label %118, label %89
#i18B

	full_text


i1 %88
6sub8B-
+
	full_text

%90 = sub nsw i64 %87, %43
%i648B

	full_text
	
i64 %87
%i648B

	full_text
	
i64 %43
8trunc8B-
+
	full_text

%91 = trunc i64 %90 to i32
%i648B

	full_text
	
i64 %90
1mul8B(
&
	full_text

%92 = mul i32 %91, 65
%i328B

	full_text
	
i32 %91
6add8B-
+
	full_text

%93 = add nsw i32 %92, %16
%i328B

	full_text
	
i32 %92
%i328B

	full_text
	
i32 %16
6sext8B,
*
	full_text

%94 = sext i32 %93 to i64
%i328B

	full_text
	
i32 %93
Xgetelementptr8BE
C
	full_text6
4
2%95 = getelementptr inbounds i32, i32* %3, i64 %94
%i648B

	full_text
	
i64 %94
Hload8B>
<
	full_text/
-
+%96 = load i32, i32* %95, align 4, !tbaa !8
'i32*8B

	full_text


i32* %95
0shl8B'
%
	full_text

%97 = shl i32 %91, 6
%i328B

	full_text
	
i32 %91
6add8B-
+
	full_text

%98 = add nsw i32 %97, %16
%i328B

	full_text
	
i32 %97
%i328B

	full_text
	
i32 %16
6sext8B,
*
	full_text

%99 = sext i32 %98 to i64
%i328B

	full_text
	
i32 %98
Ygetelementptr8BF
D
	full_text7
5
3%100 = getelementptr inbounds i32, i32* %4, i64 %99
%i648B

	full_text
	
i64 %99
Jload8B@
>
	full_text1
/
-%101 = load i32, i32* %100, align 4, !tbaa !8
(i32*8B

	full_text

	i32* %100
8add8B/
-
	full_text 

%102 = add nsw i32 %101, %96
&i328B

	full_text


i32 %101
%i328B

	full_text
	
i32 %96
2add8B)
'
	full_text

%103 = add i32 %92, 65
%i328B

	full_text
	
i32 %92
8add8B/
-
	full_text 

%104 = add nsw i32 %103, %16
&i328B

	full_text


i32 %103
%i328B

	full_text
	
i32 %16
8sext8B.
,
	full_text

%105 = sext i32 %104 to i64
&i328B

	full_text


i32 %104
Zgetelementptr8BG
E
	full_text8
6
4%106 = getelementptr inbounds i32, i32* %3, i64 %105
&i648B

	full_text


i64 %105
Jload8B@
>
	full_text1
/
-%107 = load i32, i32* %106, align 4, !tbaa !8
(i32*8B

	full_text

	i32* %106
7sub8B.
,
	full_text

%108 = sub nsw i32 %107, %6
&i328B

	full_text


i32 %107
7add8B.
,
	full_text

%109 = add nsw i32 %92, %51
%i328B

	full_text
	
i32 %92
%i328B

	full_text
	
i32 %51
8sext8B.
,
	full_text

%110 = sext i32 %109 to i64
&i328B

	full_text


i32 %109
Zgetelementptr8BG
E
	full_text8
6
4%111 = getelementptr inbounds i32, i32* %3, i64 %110
&i648B

	full_text


i64 %110
Jload8B@
>
	full_text1
/
-%112 = load i32, i32* %111, align 4, !tbaa !8
(i32*8B

	full_text

	i32* %111
7sub8B.
,
	full_text

%113 = sub nsw i32 %112, %6
&i328B

	full_text


i32 %112
[call8BQ
O
	full_textB
@
>%114 = tail call i32 @maximum(i32 %102, i32 %108, i32 %113) #6
&i328B

	full_text


i32 %102
&i328B

	full_text


i32 %108
&i328B

	full_text


i32 %113
8add8B/
-
	full_text 

%115 = add nsw i32 %103, %51
&i328B

	full_text


i32 %103
%i328B

	full_text
	
i32 %51
8sext8B.
,
	full_text

%116 = sext i32 %115 to i64
&i328B

	full_text


i32 %115
Zgetelementptr8BG
E
	full_text8
6
4%117 = getelementptr inbounds i32, i32* %3, i64 %116
&i648B

	full_text


i64 %116
Jstore8B?
=
	full_text0
.
,store i32 %114, i32* %117, align 4, !tbaa !8
&i328B

	full_text


i32 %114
(i32*8B

	full_text

	i32* %117
(br8B 

	full_text

br label %118
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
9add8B0
.
	full_text!

%119 = add nuw nsw i64 %87, 1
%i648B

	full_text
	
i64 %87
8icmp8B.
,
	full_text

%120 = icmp eq i64 %119, 64
&i648B

	full_text


i64 %119
;br8B3
1
	full_text$
"
 br i1 %120, label %79, label %86
$i18B

	full_text
	
i1 %120
Fphi8	B=
;
	full_text.
,
*%122 = phi i64 [ 62, %79 ], [ %152, %151 ]
&i648	B

	full_text


i64 %152
:icmp8	B0
.
	full_text!

%123 = icmp slt i64 %122, %43
&i648	B

	full_text


i64 %122
%i648	B

	full_text
	
i64 %43
=br8	B5
3
	full_text&
$
"br i1 %123, label %151, label %124
$i18	B

	full_text
	
i1 %123
:trunc8
B/
-
	full_text 

%125 = trunc i64 %122 to i32
&i648
B

	full_text


i64 %122
4sub8
B+
)
	full_text

%126 = sub i32 %80, %125
%i328
B

	full_text
	
i32 %80
&i328
B

	full_text


i32 %125
7add8
B.
,
	full_text

%127 = add nsw i32 %126, -1
&i328
B

	full_text


i32 %126
8add8
B/
-
	full_text 

%128 = add nsw i32 %127, %83
&i328
B

	full_text


i32 %127
%i328
B

	full_text
	
i32 %83
8sext8
B.
,
	full_text

%129 = sext i32 %128 to i64
&i328
B

	full_text


i32 %128
Zgetelementptr8
BG
E
	full_text8
6
4%130 = getelementptr inbounds i32, i32* %3, i64 %129
&i648
B

	full_text


i64 %129
Jload8
B@
>
	full_text1
/
-%131 = load i32, i32* %130, align 4, !tbaa !8
(i32*8
B

	full_text

	i32* %130
8add8
B/
-
	full_text 

%132 = add nsw i32 %127, %84
&i328
B

	full_text


i32 %127
%i328
B

	full_text
	
i32 %84
8sext8
B.
,
	full_text

%133 = sext i32 %132 to i64
&i328
B

	full_text


i32 %132
Zgetelementptr8
BG
E
	full_text8
6
4%134 = getelementptr inbounds i32, i32* %4, i64 %133
&i648
B

	full_text


i64 %133
Jload8
B@
>
	full_text1
/
-%135 = load i32, i32* %134, align 4, !tbaa !8
(i32*8
B

	full_text

	i32* %134
9add8
B0
.
	full_text!

%136 = add nsw i32 %135, %131
&i328
B

	full_text


i32 %135
&i328
B

	full_text


i32 %131
8add8
B/
-
	full_text 

%137 = add nsw i32 %127, %85
&i328
B

	full_text


i32 %127
%i328
B

	full_text
	
i32 %85
8sext8
B.
,
	full_text

%138 = sext i32 %137 to i64
&i328
B

	full_text


i32 %137
Zgetelementptr8
BG
E
	full_text8
6
4%139 = getelementptr inbounds i32, i32* %3, i64 %138
&i648
B

	full_text


i64 %138
Jload8
B@
>
	full_text1
/
-%140 = load i32, i32* %139, align 4, !tbaa !8
(i32*8
B

	full_text

	i32* %139
7sub8
B.
,
	full_text

%141 = sub nsw i32 %140, %6
&i328
B

	full_text


i32 %140
8add8
B/
-
	full_text 

%142 = add nsw i32 %126, %83
&i328
B

	full_text


i32 %126
%i328
B

	full_text
	
i32 %83
8sext8
B.
,
	full_text

%143 = sext i32 %142 to i64
&i328
B

	full_text


i32 %142
Zgetelementptr8
BG
E
	full_text8
6
4%144 = getelementptr inbounds i32, i32* %3, i64 %143
&i648
B

	full_text


i64 %143
Jload8
B@
>
	full_text1
/
-%145 = load i32, i32* %144, align 4, !tbaa !8
(i32*8
B

	full_text

	i32* %144
7sub8
B.
,
	full_text

%146 = sub nsw i32 %145, %6
&i328
B

	full_text


i32 %145
[call8
BQ
O
	full_textB
@
>%147 = tail call i32 @maximum(i32 %136, i32 %141, i32 %146) #6
&i328
B

	full_text


i32 %136
&i328
B

	full_text


i32 %141
&i328
B

	full_text


i32 %146
8add8
B/
-
	full_text 

%148 = add nsw i32 %126, %85
&i328
B

	full_text


i32 %126
%i328
B

	full_text
	
i32 %85
8sext8
B.
,
	full_text

%149 = sext i32 %148 to i64
&i328
B

	full_text


i32 %148
Zgetelementptr8
BG
E
	full_text8
6
4%150 = getelementptr inbounds i32, i32* %3, i64 %149
&i648
B

	full_text


i64 %149
Jstore8
B?
=
	full_text0
.
,store i32 %147, i32* %150, align 4, !tbaa !8
&i328
B

	full_text


i32 %147
(i32*8
B

	full_text

	i32* %150
(br8
B 

	full_text

br label %151
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
7add8B.
,
	full_text

%152 = add nsw i64 %122, -1
&i648B

	full_text


i64 %122
7icmp8B-
+
	full_text

%153 = icmp eq i64 %122, 0
&i648B

	full_text


i64 %122
=br8B5
3
	full_text&
$
"br i1 %153, label %154, label %121
$i18B

	full_text
	
i1 %153
(br8B 

	full_text

br label %156
$ret8B

	full_text


ret void
Fphi8B=
;
	full_text.
,
*%157 = phi i64 [ 0, %154 ], [ %166, %156 ]
&i648B

	full_text


i64 %166
0or8B(
&
	full_text

%158 = or i64 %157, 1
&i648B

	full_text


i64 %157
;mul8B2
0
	full_text#
!
%159 = mul nuw nsw i64 %158, 65
&i648B

	full_text


i64 %158
8add8B/
-
	full_text 

%160 = add nsw i64 %159, %58
&i648B

	full_text


i64 %159
%i648B

	full_text
	
i64 %58
Zgetelementptr8BG
E
	full_text8
6
4%161 = getelementptr inbounds i32, i32* %3, i64 %160
&i648B

	full_text


i64 %160
Jload8B@
>
	full_text1
/
-%162 = load i32, i32* %161, align 4, !tbaa !8
(i32*8B

	full_text

	i32* %161
8mul8B/
-
	full_text 

%163 = mul nsw i64 %157, %40
&i648B

	full_text


i64 %157
%i648B

	full_text
	
i64 %40
8add8B/
-
	full_text 

%164 = add nsw i64 %163, %41
&i648B

	full_text


i64 %163
%i648B

	full_text
	
i64 %41
Zgetelementptr8BG
E
	full_text8
6
4%165 = getelementptr inbounds i32, i32* %1, i64 %164
&i648B

	full_text


i64 %164
Jstore8B?
=
	full_text0
.
,store i32 %162, i32* %165, align 4, !tbaa !8
&i328B

	full_text


i32 %162
(i32*8B

	full_text

	i32* %165
6add8B-
+
	full_text

%166 = add nsw i64 %157, 2
&i648B

	full_text


i64 %157
;mul8B2
0
	full_text#
!
%167 = mul nuw nsw i64 %166, 65
&i648B

	full_text


i64 %166
8add8B/
-
	full_text 

%168 = add nsw i64 %167, %58
&i648B

	full_text


i64 %167
%i648B

	full_text
	
i64 %58
Zgetelementptr8BG
E
	full_text8
6
4%169 = getelementptr inbounds i32, i32* %3, i64 %168
&i648B

	full_text


i64 %168
Jload8B@
>
	full_text1
/
-%170 = load i32, i32* %169, align 4, !tbaa !8
(i32*8B

	full_text

	i32* %169
8mul8B/
-
	full_text 

%171 = mul nsw i64 %158, %40
&i648B

	full_text


i64 %158
%i648B

	full_text
	
i64 %40
8add8B/
-
	full_text 

%172 = add nsw i64 %171, %41
&i648B

	full_text


i64 %171
%i648B

	full_text
	
i64 %41
Zgetelementptr8BG
E
	full_text8
6
4%173 = getelementptr inbounds i32, i32* %1, i64 %172
&i648B

	full_text


i64 %172
Jstore8B?
=
	full_text0
.
,store i32 %170, i32* %173, align 4, !tbaa !8
&i328B

	full_text


i32 %170
(i32*8B

	full_text

	i32* %173
8icmp8B.
,
	full_text

%174 = icmp eq i64 %166, 64
&i648B

	full_text


i64 %166
=br8B5
3
	full_text&
$
"br i1 %174, label %155, label %156
$i18B

	full_text
	
i1 %174
%i328B

	full_text
	
i32 %11
$i328B

	full_text


i32 %8
&i32*8B

	full_text
	
i32* %0
&i32*8B

	full_text
	
i32* %4
%i328B

	full_text
	
i32 %10
&i32*8B

	full_text
	
i32* %1
$i328B

	full_text


i32 %6
&i32*8B

	full_text
	
i32* %3
$i328B

	full_text


i32 %5
$i328B

	full_text


i32 %7
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
*i328B

	full_text

i32 67108863
#i648B

	full_text	

i64 6
$i328B

	full_text


i32 65
.i648B#
!
	full_text

i64 279172874240
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 0
#i648B

	full_text	

i64 2
$i648B

	full_text


i64 64
$i648B

	full_text


i64 -1
$i328B

	full_text


i32 64
#i648B

	full_text	

i64 1
$i648B

	full_text


i64 65
#i328B

	full_text	

i32 1
$i648B

	full_text


i64 62
#i328B

	full_text	

i32 6
$i648B

	full_text


i64 32
$i328B

	full_text


i32 -1       	 
                        !  "    #$ ## %& %% '( '* )) +, ++ -. -- /0 // 12 11 34 33 56 57 55 89 :; :: <= << >? >> @A BC BB DE DD FG FH FF IJ II KL KK MN MM OP OO QR QQ ST SS UV UU WX WY WW ZZ [\ [[ ]^ ]] _` __ ab aa cd cc ef eg ee hh ik jj lm ln ll op oq oo rs rr tu tt vw vv xy xz xx {| {{ }~ } }} ÄÅ ÄÄ ÇÉ Ç
Ñ ÇÇ ÖÜ Ö
á ÖÖ à
â àà äã ää åç åå éè é
ê éé ë
í ëë ìî ì
ï ìì ñó ññ òô òò öõ öù úú û
ü ûû †° †† ¢£ ¢¢ §• §§ ¶ß ¶¶ ®
™ ©© ´¨ ´
≠ ´´ ÆØ Æ± ∞
≤ ∞∞ ≥¥ ≥≥ µ∂ µµ ∑∏ ∑
π ∑∑ ∫ª ∫∫ º
Ω ºº æø ææ ¿¡ ¿¿ ¬√ ¬
ƒ ¬¬ ≈∆ ≈≈ «
» «« …  …… ÀÃ À
Õ ÀÀ Œœ ŒŒ –— –
“ –– ”‘ ”” ’
÷ ’’ ◊ÿ ◊◊ Ÿ⁄ ŸŸ €‹ €
› €€ ﬁﬂ ﬁﬁ ‡
· ‡‡ ‚„ ‚‚ ‰Â ‰‰ ÊÁ Ê
Ë Ê
È ÊÊ ÍÎ Í
Ï ÍÍ ÌÓ ÌÌ Ô
 ÔÔ ÒÚ Ò
Û ÒÒ Ùı ˆ˜ ˆˆ ¯˘ ¯¯ ˙˚ ˙
˝ ¸¸ ˛ˇ ˛
Ä ˛˛ ÅÇ ÅÑ ÉÉ ÖÜ Ö
á ÖÖ àâ àà äã ä
å ää çé çç è
ê èè ëí ëë ìî ì
ï ìì ñó ññ ò
ô òò öõ öö úù ú
û úú ü† ü
° üü ¢£ ¢¢ §
• §§ ¶ß ¶¶ ®© ®® ™´ ™
¨ ™™ ≠Æ ≠≠ Ø
∞ ØØ ±≤ ±± ≥¥ ≥≥ µ∂ µ
∑ µ
∏ µµ π∫ π
ª ππ ºΩ ºº æ
ø ææ ¿¡ ¿
¬ ¿¿ √ƒ ≈∆ ≈≈ «» «« …  …
Œ ÕÕ œ– œœ —“ —— ”‘ ”
’ ”” ÷
◊ ÷÷ ÿŸ ÿÿ ⁄€ ⁄
‹ ⁄⁄ ›ﬁ ›
ﬂ ›› ‡
· ‡‡ ‚„ ‚
‰ ‚‚ ÂÊ ÂÂ ÁË ÁÁ ÈÍ È
Î ÈÈ Ï
Ì ÏÏ ÓÔ ÓÓ Ò 
Ú  ÛÙ Û
ı ÛÛ ˆ
˜ ˆˆ ¯˘ ¯
˙ ¯¯ ˚¸ ˚˚ ˝˛ ˝	ˇ Ä Ä Å rÅ àÇ {Ç ëÇ «Ç ò	É Ñ +Ñ KÑ ]Ñ ‡Ñ ˆ
Ö Ÿ
Ö ‰
Ö ®
Ö ≥Ü 3Ü UÜ cÜ ºÜ ’Ü ‡Ü ÔÜ èÜ §Ü ØÜ æÜ ÷Ü Ï	á á á 9	á B	á D	à    	 
            ! " $ &% ( *) ,+ . 0/ 21 4- 63 7  ; =< ? CB ED G HF JI LK N PO RQ TS VM XU Y# \[ ^] `O ba d_ fc gñ kj m9 nl p: qo sr uj wv y> zx |t ~{ j ÅÄ É9 ÑÇ Ü: áÖ âà ãÄ çå è> êé íä îë ïj óñ ôò õ ù üû °† £† •û ßˆ ™© ¨> ≠´ Ø© ±> ≤∞ ¥≥ ∂µ ∏ π∑ ª∫ Ωº ø≥ ¡¿ √ ƒ¬ ∆≈ »«  … Ãæ Õµ œŒ — “– ‘” ÷’ ÿ◊ ⁄µ ‹O ›€ ﬂﬁ ·‡ „‚ ÂÀ ÁŸ Ë‰ ÈŒ ÎO ÏÍ ÓÌ Ê ÚÔ Û© ˜ˆ ˘¯ ˚≈ ˝¸ ˇ> Ä˛ Ç¸ Ñú ÜÉ áÖ âà ã¢ åä éç êè íà î§ ïì óñ ôò õö ùë ûà †¶ °ü £¢ •§ ß¶ ©Ö ´¢ ¨™ Æ≠ ∞Ø ≤± ¥ú ∂® ∑≥ ∏Ö ∫¶ ªπ Ωº øµ ¡æ ¬¸ ∆¸ »«  Â ŒÕ –œ “— ‘a ’” ◊÷ ŸÕ €9 ‹⁄ ﬁ: ﬂ› ·ÿ „‡ ‰Õ ÊÂ ËÁ Ía ÎÈ ÌÏ Ôœ Ò9 Ú Ù: ıÛ ˜Ó ˘ˆ ˙Â ¸˚ ˛' )' 98 9@ jö Aö ji ©Æ ıÆ ∞˙ ú˙ ©Ù ı® ¸Å ƒÅ É… À… ¸√ ƒÀ Õ˝ Ã˝ Õ Ã åå ää ãã ââ ää ƒ ãã ƒµ åå µÊ åå ÊA ãã AZ ãã Zı ãã ıh ãã h ââ 	ç 	é v
é å	è Q
è ¢
è ¶
è µ
è Œ	ê /ë ë 	ë %í jí ©
í «í Õ
ì ñ
ì Â
î ò
î ¯
î ˚
ï ≈
ñ úñ û
ó Ä
ó ˆ
ó œ
ò —
ò Á	ô 	ô #ô A	ô Oô Zô hô ıô ƒö ¸	õ 	õ 
õ §
õ ¿	ú 1	ú <	ú >
ù †
ù à"

nw_kernel2"
_Z12get_group_idj"
_Z12get_local_idj"
_Z7barrierj"	
maximum*ï
rodinia-3.1-nw-nw_kernel2.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02Ä

wgsize


devmap_label


wgsize_log1p
á·çA

transfer_bytes
åÄÉ
 
transfer_bytes_log1p
á·çA