
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
4addB-
+
	full_text

%17 = add i32 %7, 67108863
0subB)
'
	full_text

%18 = sub i32 %17, %14
#i32B

	full_text
	
i32 %17
#i32B

	full_text
	
i32 %14
.shlB'
%
	full_text

%19 = shl i32 %18, 6
#i32B

	full_text
	
i32 %18
2shlB+
)
	full_text

%20 = shl nsw i32 %14, 6
#i32B

	full_text
	
i32 %14
0addB)
'
	full_text

%21 = add i32 %19, %10
#i32B

	full_text
	
i32 %19
/mulB(
&
	full_text

%22 = mul i32 %21, %5
#i32B

	full_text
	
i32 %21
0addB)
'
	full_text

%23 = add i32 %20, %11
#i32B

	full_text
	
i32 %20
0addB)
'
	full_text

%24 = add i32 %23, %22
#i32B

	full_text
	
i32 %23
#i32B

	full_text
	
i32 %22
4addB-
+
	full_text

%25 = add nsw i32 %24, %16
#i32B

	full_text
	
i32 %24
#i32B

	full_text
	
i32 %16
1addB*
(
	full_text

%26 = add nsw i32 %5, 1
4addB-
+
	full_text

%27 = add nsw i32 %26, %25
#i32B

	full_text
	
i32 %26
#i32B

	full_text
	
i32 %25
2addB+
)
	full_text

%28 = add nsw i32 %25, 1
#i32B

	full_text
	
i32 %25
3icmpB+
)
	full_text

%29 = icmp eq i32 %16, 0
#i32B

	full_text
	
i32 %16
8brB2
0
	full_text#
!
br i1 %29, label %30, label %37
!i1B

	full_text


i1 %29
6sext8B,
*
	full_text

%31 = sext i32 %25 to i64
%i328B

	full_text
	
i32 %25
Xgetelementptr8BE
C
	full_text6
4
2%32 = getelementptr inbounds i32, i32* %1, i64 %31
%i648B

	full_text
	
i64 %31
Hload8B>
<
	full_text/
-
+%33 = load i32, i32* %32, align 4, !tbaa !8
'i32*8B

	full_text


i32* %32
;mul8B2
0
	full_text#
!
%34 = mul i64 %15, 279172874240
%i648B

	full_text
	
i64 %15
9ashr8B/
-
	full_text 

%35 = ashr exact i64 %34, 32
%i648B

	full_text
	
i64 %34
Xgetelementptr8BE
C
	full_text6
4
2%36 = getelementptr inbounds i32, i32* %3, i64 %35
%i648B

	full_text
	
i64 %35
Hstore8B=
;
	full_text.
,
*store i32 %33, i32* %36, align 4, !tbaa !8
%i328B

	full_text
	
i32 %33
'i32*8B

	full_text


i32* %36
'br8B

	full_text

br label %37
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
5sext8B+
)
	full_text

%38 = sext i32 %5 to i64
6sext8B,
*
	full_text

%39 = sext i32 %27 to i64
%i328B

	full_text
	
i32 %27
1shl8B(
&
	full_text

%40 = shl i64 %15, 32
%i648B

	full_text
	
i64 %15
9ashr8B/
-
	full_text 

%41 = ashr exact i64 %40, 32
%i648B

	full_text
	
i64 %40
'br8B

	full_text

br label %58
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
5mul8B,
*
	full_text

%43 = mul nsw i32 %16, %5
%i328B

	full_text
	
i32 %16
1add8B(
&
	full_text

%44 = add i32 %43, %5
%i328B

	full_text
	
i32 %43
2add8B)
'
	full_text

%45 = add i32 %44, %24
%i328B

	full_text
	
i32 %44
%i328B

	full_text
	
i32 %24
6sext8B,
*
	full_text

%46 = sext i32 %45 to i64
%i328B

	full_text
	
i32 %45
Xgetelementptr8BE
C
	full_text6
4
2%47 = getelementptr inbounds i32, i32* %1, i64 %46
%i648B

	full_text
	
i64 %46
Hload8B>
<
	full_text/
-
+%48 = load i32, i32* %47, align 4, !tbaa !8
'i32*8B

	full_text


i32* %47
4add8B+
)
	full_text

%49 = add nsw i32 %16, 1
%i328B

	full_text
	
i32 %16
5mul8B,
*
	full_text

%50 = mul nsw i32 %49, 65
%i328B

	full_text
	
i32 %49
6sext8B,
*
	full_text

%51 = sext i32 %50 to i64
%i328B

	full_text
	
i32 %50
Xgetelementptr8BE
C
	full_text6
4
2%52 = getelementptr inbounds i32, i32* %3, i64 %51
%i648B

	full_text
	
i64 %51
Hstore8B=
;
	full_text.
,
*store i32 %48, i32* %52, align 4, !tbaa !8
%i328B

	full_text
	
i32 %48
'i32*8B

	full_text


i32* %52
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
6sext8B,
*
	full_text

%53 = sext i32 %28 to i64
%i328B

	full_text
	
i32 %28
Xgetelementptr8BE
C
	full_text6
4
2%54 = getelementptr inbounds i32, i32* %1, i64 %53
%i648B

	full_text
	
i64 %53
Hload8B>
<
	full_text/
-
+%55 = load i32, i32* %54, align 4, !tbaa !8
'i32*8B

	full_text


i32* %54
6sext8B,
*
	full_text

%56 = sext i32 %49 to i64
%i328B

	full_text
	
i32 %49
Xgetelementptr8BE
C
	full_text6
4
2%57 = getelementptr inbounds i32, i32* %3, i64 %56
%i648B

	full_text
	
i64 %56
Hstore8B=
;
	full_text.
,
*store i32 %55, i32* %57, align 4, !tbaa !8
%i328B

	full_text
	
i32 %55
'i32*8B

	full_text


i32* %57
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
'br8B

	full_text

br label %84
Bphi8B9
7
	full_text*
(
&%59 = phi i64 [ 0, %37 ], [ %75, %58 ]
%i648B

	full_text
	
i64 %75
6mul8B-
+
	full_text

%60 = mul nsw i64 %59, %38
%i648B

	full_text
	
i64 %59
%i648B

	full_text
	
i64 %38
6add8B-
+
	full_text

%61 = add nsw i64 %60, %39
%i648B

	full_text
	
i64 %60
%i648B

	full_text
	
i64 %39
Xgetelementptr8BE
C
	full_text6
4
2%62 = getelementptr inbounds i32, i32* %0, i64 %61
%i648B

	full_text
	
i64 %61
Hload8B>
<
	full_text/
-
+%63 = load i32, i32* %62, align 4, !tbaa !8
'i32*8B

	full_text


i32* %62
0shl8B'
%
	full_text

%64 = shl i64 %59, 6
%i648B

	full_text
	
i64 %59
6add8B-
+
	full_text

%65 = add nsw i64 %64, %41
%i648B

	full_text
	
i64 %64
%i648B

	full_text
	
i64 %41
Xgetelementptr8BE
C
	full_text6
4
2%66 = getelementptr inbounds i32, i32* %4, i64 %65
%i648B

	full_text
	
i64 %65
Hstore8B=
;
	full_text.
,
*store i32 %63, i32* %66, align 4, !tbaa !8
%i328B

	full_text
	
i32 %63
'i32*8B

	full_text


i32* %66
.or8B&
$
	full_text

%67 = or i64 %59, 1
%i648B

	full_text
	
i64 %59
6mul8B-
+
	full_text

%68 = mul nsw i64 %67, %38
%i648B

	full_text
	
i64 %67
%i648B

	full_text
	
i64 %38
6add8B-
+
	full_text

%69 = add nsw i64 %68, %39
%i648B

	full_text
	
i64 %68
%i648B

	full_text
	
i64 %39
Xgetelementptr8BE
C
	full_text6
4
2%70 = getelementptr inbounds i32, i32* %0, i64 %69
%i648B

	full_text
	
i64 %69
Hload8B>
<
	full_text/
-
+%71 = load i32, i32* %70, align 4, !tbaa !8
'i32*8B

	full_text


i32* %70
0shl8B'
%
	full_text

%72 = shl i64 %67, 6
%i648B

	full_text
	
i64 %67
6add8B-
+
	full_text

%73 = add nsw i64 %72, %41
%i648B

	full_text
	
i64 %72
%i648B

	full_text
	
i64 %41
Xgetelementptr8BE
C
	full_text6
4
2%74 = getelementptr inbounds i32, i32* %4, i64 %73
%i648B

	full_text
	
i64 %73
Hstore8B=
;
	full_text.
,
*store i32 %71, i32* %74, align 4, !tbaa !8
%i328B

	full_text
	
i32 %71
'i32*8B

	full_text


i32* %74
4add8B+
)
	full_text

%75 = add nsw i64 %59, 2
%i648B

	full_text
	
i64 %59
6icmp8B,
*
	full_text

%76 = icmp eq i64 %75, 64
%i648B

	full_text
	
i64 %75
:br8B2
0
	full_text#
!
br i1 %76, label %42, label %58
#i18B

	full_text


i1 %76
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
5add8B,
*
	full_text

%78 = add nsw i32 %16, 64
%i328B

	full_text
	
i32 %16
5sub8B,
*
	full_text

%79 = sub nsw i32 64, %16
%i328B

	full_text
	
i32 %16
5add8B,
*
	full_text

%80 = add nsw i32 %79, -1
%i328B

	full_text
	
i32 %79
5mul8B,
*
	full_text

%81 = mul nsw i32 %80, 65
%i328B

	full_text
	
i32 %80
0shl8B'
%
	full_text

%82 = shl i32 %80, 6
%i328B

	full_text
	
i32 %80
5mul8B,
*
	full_text

%83 = mul nsw i32 %79, 65
%i328B

	full_text
	
i32 %79
(br8B 

	full_text

br label %119
Dphi8B;
9
	full_text,
*
(%85 = phi i64 [ 0, %42 ], [ %117, %116 ]
&i648B

	full_text


i64 %117
8icmp8B.
,
	full_text

%86 = icmp slt i64 %85, %41
%i648B

	full_text
	
i64 %85
%i648B

	full_text
	
i64 %41
;br8B3
1
	full_text$
"
 br i1 %86, label %116, label %87
#i18B

	full_text


i1 %86
6sub8B-
+
	full_text

%88 = sub nsw i64 %85, %41
%i648B

	full_text
	
i64 %85
%i648B

	full_text
	
i64 %41
8trunc8B-
+
	full_text

%89 = trunc i64 %88 to i32
%i648B

	full_text
	
i64 %88
1mul8B(
&
	full_text

%90 = mul i32 %89, 65
%i328B

	full_text
	
i32 %89
6add8B-
+
	full_text

%91 = add nsw i32 %90, %16
%i328B

	full_text
	
i32 %90
%i328B

	full_text
	
i32 %16
6sext8B,
*
	full_text

%92 = sext i32 %91 to i64
%i328B

	full_text
	
i32 %91
Xgetelementptr8BE
C
	full_text6
4
2%93 = getelementptr inbounds i32, i32* %3, i64 %92
%i648B

	full_text
	
i64 %92
Hload8B>
<
	full_text/
-
+%94 = load i32, i32* %93, align 4, !tbaa !8
'i32*8B

	full_text


i32* %93
0shl8B'
%
	full_text

%95 = shl i32 %89, 6
%i328B

	full_text
	
i32 %89
6add8B-
+
	full_text

%96 = add nsw i32 %95, %16
%i328B

	full_text
	
i32 %95
%i328B

	full_text
	
i32 %16
6sext8B,
*
	full_text

%97 = sext i32 %96 to i64
%i328B

	full_text
	
i32 %96
Xgetelementptr8BE
C
	full_text6
4
2%98 = getelementptr inbounds i32, i32* %4, i64 %97
%i648B

	full_text
	
i64 %97
Hload8B>
<
	full_text/
-
+%99 = load i32, i32* %98, align 4, !tbaa !8
'i32*8B

	full_text


i32* %98
7add8B.
,
	full_text

%100 = add nsw i32 %99, %94
%i328B

	full_text
	
i32 %99
%i328B

	full_text
	
i32 %94
2add8B)
'
	full_text

%101 = add i32 %90, 65
%i328B

	full_text
	
i32 %90
8add8B/
-
	full_text 

%102 = add nsw i32 %101, %16
&i328B

	full_text


i32 %101
%i328B

	full_text
	
i32 %16
8sext8B.
,
	full_text

%103 = sext i32 %102 to i64
&i328B

	full_text


i32 %102
Zgetelementptr8BG
E
	full_text8
6
4%104 = getelementptr inbounds i32, i32* %3, i64 %103
&i648B

	full_text


i64 %103
Jload8B@
>
	full_text1
/
-%105 = load i32, i32* %104, align 4, !tbaa !8
(i32*8B

	full_text

	i32* %104
7sub8B.
,
	full_text

%106 = sub nsw i32 %105, %6
&i328B

	full_text


i32 %105
7add8B.
,
	full_text

%107 = add nsw i32 %90, %49
%i328B

	full_text
	
i32 %90
%i328B

	full_text
	
i32 %49
8sext8B.
,
	full_text

%108 = sext i32 %107 to i64
&i328B

	full_text


i32 %107
Zgetelementptr8BG
E
	full_text8
6
4%109 = getelementptr inbounds i32, i32* %3, i64 %108
&i648B

	full_text


i64 %108
Jload8B@
>
	full_text1
/
-%110 = load i32, i32* %109, align 4, !tbaa !8
(i32*8B

	full_text

	i32* %109
7sub8B.
,
	full_text

%111 = sub nsw i32 %110, %6
&i328B

	full_text


i32 %110
[call8BQ
O
	full_textB
@
>%112 = tail call i32 @maximum(i32 %100, i32 %106, i32 %111) #6
&i328B

	full_text


i32 %100
&i328B

	full_text


i32 %106
&i328B

	full_text


i32 %111
8add8B/
-
	full_text 

%113 = add nsw i32 %101, %49
&i328B

	full_text


i32 %101
%i328B

	full_text
	
i32 %49
8sext8B.
,
	full_text

%114 = sext i32 %113 to i64
&i328B

	full_text


i32 %113
Zgetelementptr8BG
E
	full_text8
6
4%115 = getelementptr inbounds i32, i32* %3, i64 %114
&i648B

	full_text


i64 %114
Jstore8B?
=
	full_text0
.
,store i32 %112, i32* %115, align 4, !tbaa !8
&i328B

	full_text


i32 %112
(i32*8B

	full_text

	i32* %115
(br8B 

	full_text

br label %116
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
9add8B0
.
	full_text!

%117 = add nuw nsw i64 %85, 1
%i648B

	full_text
	
i64 %85
8icmp8B.
,
	full_text

%118 = icmp eq i64 %117, 64
&i648B

	full_text


i64 %117
;br8B3
1
	full_text$
"
 br i1 %118, label %77, label %84
$i18B

	full_text
	
i1 %118
Fphi8	B=
;
	full_text.
,
*%120 = phi i64 [ 62, %77 ], [ %150, %149 ]
&i648	B

	full_text


i64 %150
:icmp8	B0
.
	full_text!

%121 = icmp slt i64 %120, %41
&i648	B

	full_text


i64 %120
%i648	B

	full_text
	
i64 %41
=br8	B5
3
	full_text&
$
"br i1 %121, label %149, label %122
$i18	B

	full_text
	
i1 %121
:trunc8
B/
-
	full_text 

%123 = trunc i64 %120 to i32
&i648
B

	full_text


i64 %120
4sub8
B+
)
	full_text

%124 = sub i32 %78, %123
%i328
B

	full_text
	
i32 %78
&i328
B

	full_text


i32 %123
7add8
B.
,
	full_text

%125 = add nsw i32 %124, -1
&i328
B

	full_text


i32 %124
8add8
B/
-
	full_text 

%126 = add nsw i32 %125, %81
&i328
B

	full_text


i32 %125
%i328
B

	full_text
	
i32 %81
8sext8
B.
,
	full_text

%127 = sext i32 %126 to i64
&i328
B

	full_text


i32 %126
Zgetelementptr8
BG
E
	full_text8
6
4%128 = getelementptr inbounds i32, i32* %3, i64 %127
&i648
B

	full_text


i64 %127
Jload8
B@
>
	full_text1
/
-%129 = load i32, i32* %128, align 4, !tbaa !8
(i32*8
B

	full_text

	i32* %128
8add8
B/
-
	full_text 

%130 = add nsw i32 %125, %82
&i328
B

	full_text


i32 %125
%i328
B

	full_text
	
i32 %82
8sext8
B.
,
	full_text

%131 = sext i32 %130 to i64
&i328
B

	full_text


i32 %130
Zgetelementptr8
BG
E
	full_text8
6
4%132 = getelementptr inbounds i32, i32* %4, i64 %131
&i648
B

	full_text


i64 %131
Jload8
B@
>
	full_text1
/
-%133 = load i32, i32* %132, align 4, !tbaa !8
(i32*8
B

	full_text

	i32* %132
9add8
B0
.
	full_text!

%134 = add nsw i32 %133, %129
&i328
B

	full_text


i32 %133
&i328
B

	full_text


i32 %129
8add8
B/
-
	full_text 

%135 = add nsw i32 %125, %83
&i328
B

	full_text


i32 %125
%i328
B

	full_text
	
i32 %83
8sext8
B.
,
	full_text

%136 = sext i32 %135 to i64
&i328
B

	full_text


i32 %135
Zgetelementptr8
BG
E
	full_text8
6
4%137 = getelementptr inbounds i32, i32* %3, i64 %136
&i648
B

	full_text


i64 %136
Jload8
B@
>
	full_text1
/
-%138 = load i32, i32* %137, align 4, !tbaa !8
(i32*8
B

	full_text

	i32* %137
7sub8
B.
,
	full_text

%139 = sub nsw i32 %138, %6
&i328
B

	full_text


i32 %138
8add8
B/
-
	full_text 

%140 = add nsw i32 %124, %81
&i328
B

	full_text


i32 %124
%i328
B

	full_text
	
i32 %81
8sext8
B.
,
	full_text

%141 = sext i32 %140 to i64
&i328
B

	full_text


i32 %140
Zgetelementptr8
BG
E
	full_text8
6
4%142 = getelementptr inbounds i32, i32* %3, i64 %141
&i648
B

	full_text


i64 %141
Jload8
B@
>
	full_text1
/
-%143 = load i32, i32* %142, align 4, !tbaa !8
(i32*8
B

	full_text

	i32* %142
7sub8
B.
,
	full_text

%144 = sub nsw i32 %143, %6
&i328
B

	full_text


i32 %143
[call8
BQ
O
	full_textB
@
>%145 = tail call i32 @maximum(i32 %134, i32 %139, i32 %144) #6
&i328
B

	full_text


i32 %134
&i328
B

	full_text


i32 %139
&i328
B

	full_text


i32 %144
8add8
B/
-
	full_text 

%146 = add nsw i32 %124, %83
&i328
B

	full_text


i32 %124
%i328
B

	full_text
	
i32 %83
8sext8
B.
,
	full_text

%147 = sext i32 %146 to i64
&i328
B

	full_text


i32 %146
Zgetelementptr8
BG
E
	full_text8
6
4%148 = getelementptr inbounds i32, i32* %3, i64 %147
&i648
B

	full_text


i64 %147
Jstore8
B?
=
	full_text0
.
,store i32 %145, i32* %148, align 4, !tbaa !8
&i328
B

	full_text


i32 %145
(i32*8
B

	full_text

	i32* %148
(br8
B 

	full_text

br label %149
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
7add8B.
,
	full_text

%150 = add nsw i64 %120, -1
&i648B

	full_text


i64 %120
7icmp8B-
+
	full_text

%151 = icmp eq i64 %120, 0
&i648B

	full_text


i64 %120
=br8B5
3
	full_text&
$
"br i1 %151, label %152, label %119
$i18B

	full_text
	
i1 %151
(br8B 

	full_text

br label %154
$ret8B

	full_text


ret void
Fphi8B=
;
	full_text.
,
*%155 = phi i64 [ 0, %152 ], [ %164, %154 ]
&i648B

	full_text


i64 %164
0or8B(
&
	full_text

%156 = or i64 %155, 1
&i648B

	full_text


i64 %155
;mul8B2
0
	full_text#
!
%157 = mul nuw nsw i64 %156, 65
&i648B

	full_text


i64 %156
8add8B/
-
	full_text 

%158 = add nsw i64 %157, %56
&i648B

	full_text


i64 %157
%i648B

	full_text
	
i64 %56
Zgetelementptr8BG
E
	full_text8
6
4%159 = getelementptr inbounds i32, i32* %3, i64 %158
&i648B

	full_text


i64 %158
Jload8B@
>
	full_text1
/
-%160 = load i32, i32* %159, align 4, !tbaa !8
(i32*8B

	full_text

	i32* %159
8mul8B/
-
	full_text 

%161 = mul nsw i64 %155, %38
&i648B

	full_text


i64 %155
%i648B

	full_text
	
i64 %38
8add8B/
-
	full_text 

%162 = add nsw i64 %161, %39
&i648B

	full_text


i64 %161
%i648B

	full_text
	
i64 %39
Zgetelementptr8BG
E
	full_text8
6
4%163 = getelementptr inbounds i32, i32* %1, i64 %162
&i648B

	full_text


i64 %162
Jstore8B?
=
	full_text0
.
,store i32 %160, i32* %163, align 4, !tbaa !8
&i328B

	full_text


i32 %160
(i32*8B

	full_text

	i32* %163
6add8B-
+
	full_text

%164 = add nsw i64 %155, 2
&i648B

	full_text


i64 %155
;mul8B2
0
	full_text#
!
%165 = mul nuw nsw i64 %164, 65
&i648B

	full_text


i64 %164
8add8B/
-
	full_text 

%166 = add nsw i64 %165, %56
&i648B

	full_text


i64 %165
%i648B

	full_text
	
i64 %56
Zgetelementptr8BG
E
	full_text8
6
4%167 = getelementptr inbounds i32, i32* %3, i64 %166
&i648B

	full_text


i64 %166
Jload8B@
>
	full_text1
/
-%168 = load i32, i32* %167, align 4, !tbaa !8
(i32*8B

	full_text

	i32* %167
8mul8B/
-
	full_text 

%169 = mul nsw i64 %156, %38
&i648B

	full_text


i64 %156
%i648B

	full_text
	
i64 %38
8add8B/
-
	full_text 

%170 = add nsw i64 %169, %39
&i648B

	full_text


i64 %169
%i648B

	full_text
	
i64 %39
Zgetelementptr8BG
E
	full_text8
6
4%171 = getelementptr inbounds i32, i32* %1, i64 %170
&i648B

	full_text


i64 %170
Jstore8B?
=
	full_text0
.
,store i32 %168, i32* %171, align 4, !tbaa !8
&i328B

	full_text


i32 %168
(i32*8B

	full_text

	i32* %171
8icmp8B.
,
	full_text

%172 = icmp eq i64 %164, 64
&i648B

	full_text


i64 %164
=br8B5
3
	full_text&
$
"br i1 %172, label %153, label %154
$i18B

	full_text
	
i1 %172
&i32*8B

	full_text
	
i32* %1
%i328B

	full_text
	
i32 %10
$i328B

	full_text


i32 %5
&i32*8B

	full_text
	
i32* %3
&i32*8B

	full_text
	
i32* %4
$i328B

	full_text


i32 %6
%i328B

	full_text
	
i32 %11
&i32*8B

	full_text
	
i32* %0
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
$i648B

	full_text


i64 65
#i648B

	full_text	

i64 1
#i648B

	full_text	

i64 6
$i328B

	full_text


i32 64
#i648B

	full_text	

i64 2
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
#i328B

	full_text	

i32 1
$i328B

	full_text


i32 -1
$i648B

	full_text


i64 32
*i328B

	full_text

i32 67108863
$i328B

	full_text


i32 65
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
#i328B

	full_text	

i32 6
#i648B

	full_text	

i64 0
$i648B

	full_text


i64 62       	 
                         !" !! #$ #& %% '( '' )* )) +, ++ -. -- /0 // 12 13 11 45 66 78 77 9: 99 ;< ;; => ?@ ?? AB AA CD CE CC FG FF HI HH JK JJ LM LL NO NN PQ PP RS RR TU TV TT WW XY XX Z[ ZZ \] \\ ^_ ^^ `a `` bc bd bb ee fh gg ij ik ii lm ln ll op oo qr qq st ss uv uw uu xy xx z{ z| zz }~ }} Ä 	Å  ÇÉ Ç
Ñ ÇÇ Ö
Ü ÖÖ áà áá âä ââ ãå ã
ç ãã é
è éé êë ê
í êê ìî ìì ïñ ïï óò óô öõ öö ú
ù úú ûü ûû †° †† ¢£ ¢¢ §• §§ ¶
® ßß ©™ ©
´ ©© ¨≠ ¨Ø Æ
∞ ÆÆ ±≤ ±± ≥¥ ≥≥ µ∂ µ
∑ µµ ∏π ∏∏ ∫
ª ∫∫ ºΩ ºº æø ææ ¿¡ ¿
¬ ¿¿ √ƒ √√ ≈
∆ ≈≈ «» «« …  …
À …… ÃÕ ÃÃ Œœ Œ
– ŒŒ —“ —— ”
‘ ”” ’÷ ’’ ◊ÿ ◊◊ Ÿ⁄ Ÿ
€ ŸŸ ‹› ‹‹ ﬁ
ﬂ ﬁﬁ ‡· ‡‡ ‚„ ‚‚ ‰Â ‰
Ê ‰
Á ‰‰ ËÈ Ë
Í ËË ÎÏ ÎÎ Ì
Ó ÌÌ Ô Ô
Ò ÔÔ ÚÛ Ùı ÙÙ ˆ˜ ˆˆ ¯˘ ¯
˚ ˙˙ ¸˝ ¸
˛ ¸¸ ˇÄ ˇÇ ÅÅ ÉÑ É
Ö ÉÉ Üá ÜÜ àâ à
ä àà ãå ãã ç
é çç èê èè ëí ë
ì ëë îï îî ñ
ó ññ òô òò öõ ö
ú öö ùû ù
ü ùù †° †† ¢
£ ¢¢ §• §§ ¶ß ¶¶ ®© ®
™ ®® ´¨ ´´ ≠
Æ ≠≠ Ø∞ ØØ ±≤ ±± ≥¥ ≥
µ ≥
∂ ≥≥ ∑∏ ∑
π ∑∑ ∫ª ∫∫ º
Ω ºº æø æ
¿ ææ ¡¬ √ƒ √√ ≈∆ ≈≈ «» «
Ã ÀÀ ÕŒ ÕÕ œ– œœ —“ —
” —— ‘
’ ‘‘ ÷◊ ÷÷ ÿŸ ÿ
⁄ ÿÿ €‹ €
› €€ ﬁ
ﬂ ﬁﬁ ‡· ‡
‚ ‡‡ „‰ „„ ÂÊ ÂÂ ÁË Á
È ÁÁ Í
Î ÍÍ ÏÌ ÏÏ ÓÔ Ó
 ÓÓ ÒÚ Ò
Û ÒÒ Ù
ı ÙÙ ˆ˜ ˆ
¯ ˆˆ ˘˙ ˘˘ ˚¸ ˚˝ '˝ H˝ Z˝ ﬁ˝ Ù	˛ 	ˇ ˇ ˇ 6	ˇ ?	ˇ AÄ /Ä RÄ `Ä ∫Ä ”Ä ﬁÄ ÌÄ çÄ ¢Ä ≠Ä ºÄ ‘Ä ÍÅ xÅ éÅ ≈Å ñ
Ç ◊
Ç ‚
Ç ¶
Ç ±	É Ñ oÑ ÖÖ    	 
              "! $ &% (' * ,+ .- 0) 2/ 3 8 :9 < @? BA D EC GF IH K ML ON QP SJ UR V YX [Z ]L _^ a\ c` dì hg j6 ki m7 nl po rg ts v; wu yq {x |g ~} Ä6 Å É7 ÑÇ ÜÖ à} äâ å; çã èá ëé íg îì ñï ò õ ùú üû °û £ú •Ù ®ß ™; ´© ≠ß Ø; ∞Æ ≤± ¥≥ ∂ ∑µ π∏ ª∫ Ω± øæ ¡ ¬¿ ƒ√ ∆≈ »«  º À≥ ÕÃ œ –Œ “— ‘” ÷’ ÿ≥ ⁄L €Ÿ ›‹ ﬂﬁ ·‡ „… Â◊ Ê‚ ÁÃ ÈL ÍË ÏÎ Ó‰ Ì Òß ıÙ ˜ˆ ˘√ ˚˙ ˝; ˛¸ Ä˙ Çö ÑÅ ÖÉ áÜ â† äà åã éç êÜ í¢ ìë ïî óñ ôò õè úÜ û§ üù °† £¢ •§ ßÉ ©† ™® ¨´ Æ≠ ∞Ø ≤ö ¥¶ µ± ∂É ∏§ π∑ ª∫ Ω≥ øº ¿˙ ƒ˙ ∆≈ »„ ÃÀ ŒÕ –œ “^ ”— ’‘ ◊À Ÿ6 ⁄ÿ ‹7 ›€ ﬂ÷ ·ﬁ ‚À ‰„ ÊÂ Ë^ ÈÁ ÎÍ ÌÕ Ô6 Ó Ú7 ÛÒ ıÏ ˜Ù ¯„ ˙˘ ¸# %# 54 5= gó >ó gf ß¨ Û¨ Æ¯ ô¯ ßÚ Û¶ ˙ˇ ¬ˇ Å« …« ˙¡ ¬… À˚  ˚ À àà ââ ÜÜ   ááW àà Wô àà ô áá Û àà Û¬ àà ¬5 àà 5‰ ââ ‰ ÜÜ ≥ ââ ≥e àà e> àà >
ä œ
ä Â	ã }
ã Ù
ã Õ	å s
å â
ç öç ú
é ì
é „	è +ê ê 	ê !	ë 	ë ë 5ë >	ë Lë Wë eë ôë Ûë ¬
í û
í Ü	ì -	ì 9	ì ;	î 	ï N
ï †
ï §
ï ≥
ï Ã
ñ ï
ñ ˆ
ñ ˘
ó √	ò 	ò 
ò ¢
ò æô gô ß
ô ≈ô Àö ˙"

nw_kernel1"
_Z12get_group_idj"
_Z12get_local_idj"
_Z7barrierj"	
maximum*ï
rodinia-3.1-nw-nw_kernel1.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02Ä

wgsize_log1p
á·çA
 
transfer_bytes_log1p
á·çA

wgsize


transfer_bytes
åÄÉ

devmap_label
