
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
%8 = icmp slt i32 %7, %4
"i32B

	full_text


i32 %7
6brB0
.
	full_text!

br i1 %8, label %9, label %18
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
Vgetelementptr8BC
A
	full_text4
2
0%12 = getelementptr inbounds i8, i8* %1, i64 %11
%i648B

	full_text
	
i64 %11
Fload8B<
:
	full_text-
+
)%13 = load i8, i8* %12, align 1, !tbaa !8
%i8*8B

	full_text
	
i8* %12
4icmp8B*
(
	full_text

%14 = icmp eq i8 %13, 0
#i88B

	full_text


i8 %13
:br8B2
0
	full_text#
!
br i1 %14, label %18, label %15
#i18B

	full_text


i1 %14
Vgetelementptr8BC
A
	full_text4
2
0%16 = getelementptr inbounds i8, i8* %0, i64 %11
%i648B

	full_text
	
i64 %11
Dstore8B9
7
	full_text*
(
&store i8 1, i8* %16, align 1, !tbaa !8
%i8*8B

	full_text
	
i8* %16
Vgetelementptr8BC
A
	full_text4
2
0%17 = getelementptr inbounds i8, i8* %2, i64 %11
%i648B

	full_text
	
i64 %11
Dstore8B9
7
	full_text*
(
&store i8 1, i8* %17, align 1, !tbaa !8
%i8*8B

	full_text
	
i8* %17
Cstore8B8
6
	full_text)
'
%store i8 1, i8* %3, align 1, !tbaa !8
Dstore8B9
7
	full_text*
(
&store i8 0, i8* %12, align 1, !tbaa !8
%i8*8B

	full_text
	
i8* %12
'br8B

	full_text

br label %18
$ret8B

	full_text


ret void
$i8*8B

	full_text


i8* %2
$i8*8B

	full_text


i8* %1
$i328B

	full_text


i32 %4
$i8*8B

	full_text


i8* %3
$i8*8B

	full_text


i8* %0
-; undefined function B

	full_text

 
!i88B

	full_text

i8 0
!i88B

	full_text

i8 1
#i328B

	full_text	

i32 0
$i648B

	full_text


i64 32      	  
 

                   ! " # $ %     	 
    
  
              && && ' ' ( ( ( ) * * 
"
BFS_2"
_Z13get_global_idj*?
rodinia-3.1-bfs-BFS_2.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282?
 
transfer_bytes_log1p
"??A

devmap_label
 

transfer_bytes
???

wgsize_log1p
"??A

wgsize
?