

[external]
KcallBC
A
	full_text4
2
0%8 = tail call i64 @_Z13get_global_idj(i32 0) #2
4truncB+
)
	full_text

%9 = trunc i64 %8 to i32
"i64B

	full_text


i64 %8
4icmpB,
*
	full_text

%10 = icmp slt i32 %9, %6
"i32B

	full_text


i32 %9
8brB2
0
	full_text#
!
br i1 %10, label %11, label %50
!i1B

	full_text


i1 %10
0shl8B'
%
	full_text

%12 = shl i64 %8, 32
$i648B

	full_text


i64 %8
9ashr8B/
-
	full_text 

%13 = ashr exact i64 %12, 32
%i648B

	full_text
	
i64 %12
Vgetelementptr8BC
A
	full_text4
2
0%14 = getelementptr inbounds i8, i8* %2, i64 %13
%i648B

	full_text
	
i64 %13
Fload8B<
:
	full_text-
+
)%15 = load i8, i8* %14, align 1, !tbaa !8
%i8*8B

	full_text
	
i8* %14
4icmp8B*
(
	full_text

%16 = icmp eq i8 %15, 0
#i88B

	full_text


i8 %15
:br8B2
0
	full_text#
!
br i1 %16, label %50, label %17
#i18B

	full_text


i1 %16
Dstore8B9
7
	full_text*
(
&store i8 0, i8* %14, align 1, !tbaa !8
%i8*8B

	full_text
	
i8* %14
qgetelementptr8B^
\
	full_textO
M
K%18 = getelementptr inbounds %struct.Node, %struct.Node* %0, i64 %13, i32 0
%i648B

	full_text
	
i64 %13
qgetelementptr8B^
\
	full_textO
M
K%19 = getelementptr inbounds %struct.Node, %struct.Node* %0, i64 %13, i32 1
%i648B

	full_text
	
i64 %13
Iload8B?
=
	full_text0
.
,%20 = load i32, i32* %19, align 4, !tbaa !11
'i32*8B

	full_text


i32* %19
6icmp8B,
*
	full_text

%21 = icmp sgt i32 %20, 0
%i328B

	full_text
	
i32 %20
:br8B2
0
	full_text#
!
br i1 %21, label %22, label %50
#i18B

	full_text


i1 %21
Iload8B?
=
	full_text0
.
,%23 = load i32, i32* %18, align 4, !tbaa !14
'i32*8B

	full_text


i32* %18
Xgetelementptr8BE
C
	full_text6
4
2%24 = getelementptr inbounds i32, i32* %5, i64 %13
%i648B

	full_text
	
i64 %13
6sext8B,
*
	full_text

%25 = sext i32 %23 to i64
%i328B

	full_text
	
i32 %23
'br8B

	full_text

br label %26
Dphi8B;
9
	full_text,
*
(%27 = phi i32 [ %23, %22 ], [ %44, %43 ]
%i328B

	full_text
	
i32 %23
%i328B

	full_text
	
i32 %44
Dphi8B;
9
	full_text,
*
(%28 = phi i32 [ %20, %22 ], [ %45, %43 ]
%i328B

	full_text
	
i32 %20
%i328B

	full_text
	
i32 %45
Dphi8B;
9
	full_text,
*
(%29 = phi i64 [ %25, %22 ], [ %46, %43 ]
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %46
Xgetelementptr8BE
C
	full_text6
4
2%30 = getelementptr inbounds i32, i32* %1, i64 %29
%i648B

	full_text
	
i64 %29
Iload8B?
=
	full_text0
.
,%31 = load i32, i32* %30, align 4, !tbaa !15
'i32*8B

	full_text


i32* %30
6sext8B,
*
	full_text

%32 = sext i32 %31 to i64
%i328B

	full_text
	
i32 %31
Vgetelementptr8BC
A
	full_text4
2
0%33 = getelementptr inbounds i8, i8* %4, i64 %32
%i648B

	full_text
	
i64 %32
Fload8B<
:
	full_text-
+
)%34 = load i8, i8* %33, align 1, !tbaa !8
%i8*8B

	full_text
	
i8* %33
4icmp8B*
(
	full_text

%35 = icmp eq i8 %34, 0
#i88B

	full_text


i8 %34
:br8B2
0
	full_text#
!
br i1 %35, label %36, label %43
#i18B

	full_text


i1 %35
Iload8B?
=
	full_text0
.
,%37 = load i32, i32* %24, align 4, !tbaa !15
'i32*8B

	full_text


i32* %24
4add8B+
)
	full_text

%38 = add nsw i32 %37, 1
%i328B

	full_text
	
i32 %37
Xgetelementptr8BE
C
	full_text6
4
2%39 = getelementptr inbounds i32, i32* %5, i64 %32
%i648B

	full_text
	
i64 %32
Istore8B>
<
	full_text/
-
+store i32 %38, i32* %39, align 4, !tbaa !15
%i328B

	full_text
	
i32 %38
'i32*8B

	full_text


i32* %39
Vgetelementptr8BC
A
	full_text4
2
0%40 = getelementptr inbounds i8, i8* %3, i64 %32
%i648B

	full_text
	
i64 %32
Dstore8B9
7
	full_text*
(
&store i8 1, i8* %40, align 1, !tbaa !8
%i8*8B

	full_text
	
i8* %40
Iload8B?
=
	full_text0
.
,%41 = load i32, i32* %19, align 4, !tbaa !11
'i32*8B

	full_text


i32* %19
Iload8B?
=
	full_text0
.
,%42 = load i32, i32* %18, align 4, !tbaa !14
'i32*8B

	full_text


i32* %18
'br8B

	full_text

br label %43
Dphi8B;
9
	full_text,
*
(%44 = phi i32 [ %27, %26 ], [ %42, %36 ]
%i328B

	full_text
	
i32 %27
%i328B

	full_text
	
i32 %42
Dphi8B;
9
	full_text,
*
(%45 = phi i32 [ %28, %26 ], [ %41, %36 ]
%i328B

	full_text
	
i32 %28
%i328B

	full_text
	
i32 %41
0add8B'
%
	full_text

%46 = add i64 %29, 1
%i648B

	full_text
	
i64 %29
6add8B-
+
	full_text

%47 = add nsw i32 %44, %45
%i328B

	full_text
	
i32 %44
%i328B

	full_text
	
i32 %45
6sext8B,
*
	full_text

%48 = sext i32 %47 to i64
%i328B

	full_text
	
i32 %47
8icmp8B.
,
	full_text

%49 = icmp slt i64 %46, %48
%i648B

	full_text
	
i64 %46
%i648B

	full_text
	
i64 %48
:br8B2
0
	full_text#
!
br i1 %49, label %26, label %50
#i18B

	full_text


i1 %49
$ret8B

	full_text


ret void
&i32*8B

	full_text
	
i32* %5
2struct*8B#
!
	full_text

%struct.Node* %0
$i328B

	full_text


i32 %6
$i8*8B

	full_text


i8* %4
&i32*8B

	full_text
	
i32* %1
$i8*8B

	full_text


i8* %3
$i8*8B

	full_text


i8* %2
-; undefined function B

	full_text

 
!i88B

	full_text

i8 0
#i328B

	full_text	

i32 1
#i648B

	full_text	

i64 1
$i648B

	full_text


i64 32
#i328B

	full_text	

i32 0
!i88B

	full_text

i8 1      	  
 

                   !    "# "" $% $$ &( ') '' *+ *, ** -. -/ -- 01 00 23 22 45 44 67 66 89 88 :; :: <= <? >> @A @@ BC BB DE DF DD GH GG IJ II KL KK MN MM OQ PR PP ST SU SS VW VV XY XZ XX [\ [[ ]^ ]_ ]] `a `c "c Bd d e f 6g 0h Gi     	 
     
 
     !
 #  %  (P ) +S ,$ .V /- 10 32 54 76 98 ;: =" ?> A4 C@ EB F4 HG J L N' QM R* TK U- WP YS ZX \V ^[ _] a  b b    b& '< >< PO P` '` b jj b jj k k k :l l @m Vn n 
o o o p I"
BFS_1"
_Z13get_global_idj*?
rodinia-3.1-bfs-BFS_1.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02?
 
transfer_bytes_log1p
"??A

transfer_bytes
???

wgsize
?

wgsize_log1p
"??A

devmap_label
 