

[external]
KcallBC
A
	full_text4
2
0%9 = tail call i64 @_Z13get_global_idj(i32 0) #2
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
1sremB)
'
	full_text

%11 = srem i32 %10, %3
#i32B

	full_text
	
i32 %10
1sdivB)
'
	full_text

%12 = sdiv i32 %10, %3
#i32B

	full_text
	
i32 %10
3mulB,
*
	full_text

%13 = mul nsw i32 %12, %4
#i32B

	full_text
	
i32 %12
1addB*
(
	full_text

%14 = add nsw i32 %4, 1
3addB,
*
	full_text

%15 = add nsw i32 %13, %4
#i32B

	full_text
	
i32 %13
5icmpB-
+
	full_text

%16 = icmp ult i32 %15, %5
#i32B

	full_text
	
i32 %15
8brB2
0
	full_text#
!
br i1 %16, label %22, label %17
!i1B

	full_text


i1 %16
1sub8B(
&
	full_text

%18 = sub i32 %5, %13
%i328B

	full_text
	
i32 %13
0add8B'
%
	full_text

%19 = add i32 %18, 1
%i328B

	full_text
	
i32 %18
6icmp8B,
*
	full_text

%20 = icmp sgt i32 %19, 0
%i328B

	full_text
	
i32 %19
Bselect8B6
4
	full_text'
%
#%21 = select i1 %20, i32 %19, i32 0
#i18B

	full_text


i1 %20
%i328B

	full_text
	
i32 %19
'br8B

	full_text

br label %22
Cphi8B:
8
	full_text+
)
'%23 = phi i32 [ %21, %17 ], [ %14, %8 ]
%i328B

	full_text
	
i32 %21
%i328B

	full_text
	
i32 %14
1add8B(
&
	full_text

%24 = add i32 %13, -1
%i328B

	full_text
	
i32 %13
2add8B)
'
	full_text

%25 = add i32 %24, %23
%i328B

	full_text
	
i32 %24
%i328B

	full_text
	
i32 %23
8icmp8B.
,
	full_text

%26 = icmp slt i32 %13, %25
%i328B

	full_text
	
i32 %13
%i328B

	full_text
	
i32 %25
:br8B2
0
	full_text#
!
br i1 %26, label %27, label %31
#i18B

	full_text


i1 %26
3add8B*
(
	full_text

%28 = add nsw i32 %6, 1
6sext8B,
*
	full_text

%29 = sext i32 %13 to i64
%i328B

	full_text
	
i32 %13
6sext8B,
*
	full_text

%30 = sext i32 %25 to i64
%i328B

	full_text
	
i32 %25
'br8B

	full_text

br label %32
$ret8B

	full_text


ret void
Dphi8B;
9
	full_text,
*
(%33 = phi i64 [ %29, %27 ], [ %37, %60 ]
%i648B

	full_text
	
i64 %29
%i648B

	full_text
	
i64 %37
Xgetelementptr8BE
C
	full_text6
4
2%34 = getelementptr inbounds i32, i32* %0, i64 %33
%i648B

	full_text
	
i64 %33
Hload8B>
<
	full_text/
-
+%35 = load i32, i32* %34, align 4, !tbaa !8
'i32*8B

	full_text


i32* %34
6icmp8B,
*
	full_text

%36 = icmp eq i32 %35, %6
%i328B

	full_text
	
i32 %35
4add8B+
)
	full_text

%37 = add nsw i64 %33, 1
%i648B

	full_text
	
i64 %33
:br8B2
0
	full_text#
!
br i1 %36, label %38, label %60
#i18B

	full_text


i1 %36
Xgetelementptr8BE
C
	full_text6
4
2%39 = getelementptr inbounds i32, i32* %1, i64 %37
%i648B

	full_text
	
i64 %37
Hload8B>
<
	full_text/
-
+%40 = load i32, i32* %39, align 4, !tbaa !8
'i32*8B

	full_text


i32* %39
Xgetelementptr8BE
C
	full_text6
4
2%41 = getelementptr inbounds i32, i32* %1, i64 %33
%i648B

	full_text
	
i64 %33
Hload8B>
<
	full_text/
-
+%42 = load i32, i32* %41, align 4, !tbaa !8
'i32*8B

	full_text


i32* %41
2sub8B)
'
	full_text

%43 = sub i32 %40, %42
%i328B

	full_text
	
i32 %40
%i328B

	full_text
	
i32 %42
8icmp8B.
,
	full_text

%44 = icmp ult i32 %11, %43
%i328B

	full_text
	
i32 %11
%i328B

	full_text
	
i32 %43
:br8B2
0
	full_text#
!
br i1 %44, label %45, label %60
#i18B

	full_text


i1 %44
'br8B

	full_text

br label %46
Dphi8B;
9
	full_text,
*
(%47 = phi i32 [ %58, %57 ], [ %11, %45 ]
%i328B

	full_text
	
i32 %58
%i328B

	full_text
	
i32 %11
2add8B)
'
	full_text

%48 = add i32 %47, %42
%i328B

	full_text
	
i32 %47
%i328B

	full_text
	
i32 %42
6zext8B,
*
	full_text

%49 = zext i32 %48 to i64
%i328B

	full_text
	
i32 %48
Xgetelementptr8BE
C
	full_text6
4
2%50 = getelementptr inbounds i32, i32* %2, i64 %49
%i648B

	full_text
	
i64 %49
Hload8B>
<
	full_text/
-
+%51 = load i32, i32* %50, align 4, !tbaa !8
'i32*8B

	full_text


i32* %50
6sext8B,
*
	full_text

%52 = sext i32 %51 to i64
%i328B

	full_text
	
i32 %51
Xgetelementptr8BE
C
	full_text6
4
2%53 = getelementptr inbounds i32, i32* %0, i64 %52
%i648B

	full_text
	
i64 %52
Hload8B>
<
	full_text/
-
+%54 = load i32, i32* %53, align 4, !tbaa !8
'i32*8B

	full_text


i32* %53
6icmp8B,
*
	full_text

%55 = icmp eq i32 %54, -1
%i328B

	full_text
	
i32 %54
:br8B2
0
	full_text#
!
br i1 %55, label %56, label %57
#i18B

	full_text


i1 %55
Hstore8	B=
;
	full_text.
,
*store i32 %28, i32* %53, align 4, !tbaa !8
%i328	B

	full_text
	
i32 %28
'i32*8	B

	full_text


i32* %53
Estore8	B:
8
	full_text+
)
'store i32 1, i32* %7, align 4, !tbaa !8
'br8	B

	full_text

br label %57
5add8
B,
*
	full_text

%58 = add nsw i32 %47, %3
%i328
B

	full_text
	
i32 %47
8icmp8
B.
,
	full_text

%59 = icmp ult i32 %58, %43
%i328
B

	full_text
	
i32 %58
%i328
B

	full_text
	
i32 %43
:br8
B2
0
	full_text#
!
br i1 %59, label %46, label %60
#i18
B

	full_text


i1 %59
7icmp8B-
+
	full_text

%61 = icmp eq i64 %37, %30
%i648B

	full_text
	
i64 %37
%i648B

	full_text
	
i64 %30
:br8B2
0
	full_text#
!
br i1 %61, label %31, label %32
#i18B

	full_text


i1 %61
&i32*8B

	full_text
	
i32* %0
&i32*8B

	full_text
	
i32* %1
$i328B

	full_text


i32 %5
&i32*8B

	full_text
	
i32* %7
&i32*8B

	full_text
	
i32* %2
$i328B

	full_text


i32 %4
$i328B

	full_text


i32 %6
$i328B

	full_text


i32 %3
-; undefined function B

	full_text

 
#i328B

	full_text	

i32 0
#i328B

	full_text	

i32 1
$i328B

	full_text


i32 -1
#i648B

	full_text	

i64 1       	  

                     !  "    #$ #% ## &' &( )* )) +, ++ -0 /1 // 23 22 45 44 67 66 89 88 :; := << >? >> @A @@ BC BB DE DF DD GH GI GG JK JN MO MM PQ PR PP ST SS UV UU WX WW YZ YY [\ [[ ]^ ]] _` __ ab ad ce cc ff gi hh jk jl jj mn mp oq oo rs rt 2t [u <u @v v w fx Uy y 
y z (z 6{ { { h    	         
   ! " $  %# ' *  ,) 08 1/ 32 54 7/ 96 ;8 =< ?/ A@ C> EB F HD IG Kh N OM QB RP TS VU XW ZY \[ ^] `_ b( d[ eM ih kD lj n8 p+ qo s  & (& . - /: <: oJ LJ or .r /L Ma ca hg hm Mm o . || || } } } ~ 
~ ~ (~ f  _	? 8"
BFS_kernel_warp"
_Z13get_global_idj*?
!shoc-1.1.5-BFS-BFS_kernel_warp.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02?

devmap_label
 
 
transfer_bytes_log1p
T?)A

transfer_bytes
??

wgsize
?

wgsize_log1p
T?)A